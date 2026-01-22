#include "Decoder.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <libavutil/rational.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>

#include <stdexcept>

static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    const enum AVPixelFormat* p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_CUDA)
            return *p;
    }
    throw std::runtime_error("[Decoder] Failed to get HW surface format.");
}

Decoder::Decoder(const std::string& filename, bool enable_frame_skip_, int output_width, int output_height)
    : requested_width(output_width),
      requested_height(output_height),
      enable_frame_skip(enable_frame_skip_),
      output_this_frame(true) {
    init_ffmpeg(filename);
}

Decoder::~Decoder() {
    cleanup();
}

void Decoder::init_ffmpeg(const std::string& filename) {
    AVDictionary* opts = nullptr;
    if (filename.rfind("rtsp://", 0) == 0) {
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    }

    if (avformat_open_input(&format_ctx, filename.c_str(), nullptr, &opts) != 0) {
        if (opts) {
            av_dict_free(&opts);
        }
        throw std::runtime_error("[Decoder] Could not open input file: " + filename);
    }
    if (opts) {
        av_dict_free(&opts);
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        throw std::runtime_error("[Decoder] Could not find stream info");
    }

    video_stream_idx = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        throw std::runtime_error("[Decoder] Could not find video stream");
    }

    AVStream*          stream   = format_ctx->streams[video_stream_idx];
    AVCodecParameters* codecpar = stream->codecpar;

    AVRational frame_rate = stream->avg_frame_rate;
    if (frame_rate.num == 0 || frame_rate.den == 0) {
        frame_rate = stream->r_frame_rate;
    }
    if (frame_rate.num != 0 && frame_rate.den != 0) {
        fps = av_q2d(frame_rate);
    }

    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        throw std::runtime_error("[Decoder] Codec not found");
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("[Decoder] Could not allocate codec context");
    }

    if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
        throw std::runtime_error("[Decoder] Could not copy codec params");
    }

    if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        throw std::runtime_error("[Decoder] Failed to create CUDA HW device");
    }
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    codec_ctx->get_format    = get_hw_format;

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        throw std::runtime_error("[Decoder] Could not open codec");
    }

    decode_width  = codec_ctx->width;
    decode_height = codec_ctx->height;

    if (requested_width > 0 && requested_height > 0) {
        width  = requested_width;
        height = requested_height;
    } else {
        width  = decode_width;
        height = decode_height;
    }

    frame  = av_frame_alloc();
    packet = av_packet_alloc();
}

void Decoder::cleanup() {
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) avformat_close_input(&format_ctx);
    if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
}

torch::Tensor Decoder::next_frame() {
    auto process_frame = [&](AVFrame* f) -> torch::Tensor {
        if (f->format != AV_PIX_FMT_CUDA) {
            std::cerr << "[Decoder] Frame format is not CUDA: " << f->format << std::endl;
            return torch::Tensor();
        }

        cudaStream_t stream            = c10::cuda::getCurrentCUDAStream().stream();
        NppStatus    npp_stream_status = nppSetStream(stream);
        if (npp_stream_status != NPP_SUCCESS) {
            throw std::runtime_error("[Decoder] nppSetStream failed: " + std::to_string(npp_stream_status));
        }

        auto          options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA).layout(torch::kStrided);
        torch::Tensor rgb     = torch::empty({decode_height, decode_width, 3}, options);
        const Npp8u*  pSrc[2];
        pSrc[0]           = (const Npp8u*)f->data[0];
        pSrc[1]           = (const Npp8u*)f->data[1];
        int      nSrcStep = f->linesize[0];
        Npp8u*   pDst     = rgb.data_ptr<uint8_t>();
        int      nDstStep = decode_width * 3;
        NppiSize oSizeROI;
        oSizeROI.width   = decode_width;
        oSizeROI.height  = decode_height;
        NppStatus status = nppiNV12ToRGB_8u_P2C3R(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
        if (status != NPP_SUCCESS) {
            throw std::runtime_error("[Decoder] NPP conversion failed: " + std::to_string(status));
        }

        if (width == decode_width && height == decode_height) {
            return rgb;
        }

        torch::Tensor resized    = torch::empty({height, width, 3}, options);
        const Npp8u*  pResizeSrc = rgb.data_ptr<uint8_t>();
        Npp8u*        pResizeDst = resized.data_ptr<uint8_t>();

        NppiSize srcSize;
        srcSize.width  = decode_width;
        srcSize.height = decode_height;
        int srcStep    = decode_width * 3;
        int dstStep    = width * 3;

        NppiRect srcROI;
        srcROI.x      = 0;
        srcROI.y      = 0;
        srcROI.width  = decode_width;
        srcROI.height = decode_height;

        NppiRect dstROI;
        dstROI.x      = 0;
        dstROI.y      = 0;
        dstROI.width  = width;
        dstROI.height = height;

        double xFactor = static_cast<double>(width) / static_cast<double>(decode_width);
        double yFactor = static_cast<double>(height) / static_cast<double>(decode_height);

        NppStatus resize_status = nppiResizeSqrPixel_8u_C3R(
            pResizeSrc,
            srcSize,
            srcStep,
            srcROI,
            pResizeDst,
            dstStep,
            dstROI,
            xFactor,
            yFactor,
            0.0,
            0.0,
            NPPI_INTER_LINEAR);
        if (resize_status != NPP_SUCCESS) {
            throw std::runtime_error("[Decoder] NPP resize failed: " + std::to_string(resize_status));
        }

        return resized;
    };

    if (finished) return torch::Tensor();

    while (true) {
        int ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret >= 0) {
            if (enable_frame_skip && !output_this_frame) {
                output_this_frame = true;
                av_frame_unref(frame);
                continue;
            }
            output_this_frame = enable_frame_skip ? false : true;
            torch::Tensor out = process_frame(frame);
            av_frame_unref(frame);
            return out;
        } else if (ret == AVERROR_EOF) {
            finished = true;
            return torch::Tensor();
        } else if (ret != AVERROR(EAGAIN)) {
            throw std::runtime_error("[Decoder] Error receiving frame: " + std::to_string(ret));
        }

        if (flushing) {
            finished = true;
            return torch::Tensor();
        }

        ret = av_read_frame(format_ctx, packet);
        if (ret < 0) {
            flushing = true;
            avcodec_send_packet(codec_ctx, nullptr);
            continue;
        }

        if (packet->stream_index == video_stream_idx) {
            ret = avcodec_send_packet(codec_ctx, packet);
            av_packet_unref(packet);
            if (ret < 0) throw std::runtime_error("[Decoder] Error sending packet");
        } else {
            av_packet_unref(packet);
        }
    }
}
