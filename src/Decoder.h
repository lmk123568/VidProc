#pragma once

#include <torch/extension.h>

#include <memory>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

class Decoder {
public:
    Decoder(const std::string &filename, bool enable_frame_skip = false, int output_width = 0, int output_height = 0);
    ~Decoder();

    torch::Tensor next_frame();
    int           get_width() const { return width; }
    int           get_height() const { return height; }
    double        get_fps() const { return fps; }

private:
    void init_ffmpeg(const std::string &filename);
    void cleanup();

    AVFormatContext *format_ctx       = nullptr;
    AVCodecContext  *codec_ctx        = nullptr;
    AVBufferRef     *hw_device_ctx    = nullptr;
    int              video_stream_idx = -1;

    AVFrame  *frame  = nullptr;
    AVPacket *packet = nullptr;

    int    width             = 0;
    int    height            = 0;
    int    decode_width      = 0;
    int    decode_height     = 0;
    int    requested_width   = 0;
    int    requested_height  = 0;
    double fps               = 0.0;
    bool   flushing          = false;
    bool   finished          = false;
    bool   enable_frame_skip = false;
    bool   output_this_frame = true;
};
