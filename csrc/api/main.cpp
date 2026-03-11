//
// YOLOv8 TensorRT HTTP API Server
//
// Provides a REST API for object detection using YOLOv8 TensorRT engines.
// Uses cpp-httplib for HTTP and nlohmann/json for JSON serialization.
//
#include "httplib.h"
#include "nlohmann/json.hpp"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

static const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

// Global model instance protected by a mutex for thread safety
static std::unique_ptr<YOLOv8> g_model;
static std::mutex              g_mutex;

// Inference parameters
static float g_score_thres = 0.25f;
static float g_iou_thres   = 0.65f;
static int   g_topk        = 100;
static int   g_num_labels  = 80;

void print_usage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s <engine_path> [options]\n"
            "Options:\n"
            "  --port <port>          Listen port (default: 22266)\n"
            "  --host <host>          Listen host (default: 0.0.0.0)\n"
            "  --score-thres <float>  Score threshold (default: 0.25)\n"
            "  --iou-thres <float>    IoU threshold (default: 0.65)\n"
            "  --topk <int>           Top-K results (default: 100)\n",
            prog);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string engine_path = argv[1];
    std::string host        = "0.0.0.0";
    int         port        = 22266;

    // Parse optional arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        }
        else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        }
        else if (arg == "--score-thres" && i + 1 < argc) {
            g_score_thres = std::atof(argv[++i]);
        }
        else if (arg == "--iou-thres" && i + 1 < argc) {
            g_iou_thres = std::atof(argv[++i]);
        }
        else if (arg == "--topk" && i + 1 < argc) {
            g_topk = std::atoi(argv[++i]);
        }
        else {
            print_usage(argv[0]);
            return 1;
        }
    }

    // Initialize CUDA and load the TensorRT engine
    cudaSetDevice(0);
    printf("Loading TensorRT engine: %s\n", engine_path.c_str());
    g_model = std::make_unique<YOLOv8>(engine_path);
    g_model->make_pipe(true);
    printf("Engine loaded successfully.\n");

    httplib::Server svr;

    // Health check endpoint
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json j = {{"status", "ok"}};
        res.set_content(j.dump(), "application/json");
    });

    // Detection endpoint: accepts an image file via multipart form upload
    svr.Post("/detect", [](const httplib::Request& req, httplib::Response& res) {
        // Expect a file field named "image"
        if (!req.has_file("image")) {
            json err = {{"error", "Missing 'image' field in multipart form data"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        auto file = req.get_file_value("image");
        std::vector<uchar> buf(file.content.begin(), file.content.end());
        cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);

        if (image.empty()) {
            json err = {{"error", "Failed to decode image"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Optional per-request threshold overrides via query params
        float score_thres = g_score_thres;
        float iou_thres   = g_iou_thres;
        int   topk        = g_topk;

        if (req.has_param("score_thres")) {
            score_thres = std::atof(req.get_param_value("score_thres").c_str());
        }
        if (req.has_param("iou_thres")) {
            iou_thres = std::atof(req.get_param_value("iou_thres").c_str());
        }
        if (req.has_param("topk")) {
            topk = std::atoi(req.get_param_value("topk").c_str());
        }

        std::vector<Object> objs;
        double              inference_ms = 0.0;

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_model->copy_from_Mat(image);
            auto start = std::chrono::system_clock::now();
            g_model->infer();
            auto end = std::chrono::system_clock::now();
            g_model->postprocess(objs, score_thres, iou_thres, topk, g_num_labels);
            inference_ms =
                (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        }

        // Build JSON response
        json detections = json::array();
        for (auto& obj : objs) {
            json det;
            det["label"]      = CLASS_NAMES[obj.label];
            det["label_id"]   = obj.label;
            det["confidence"] = obj.prob;
            det["bbox"]       = {
                {"x",      obj.rect.x},
                {"y",      obj.rect.y},
                {"width",  obj.rect.width},
                {"height", obj.rect.height}
            };
            detections.push_back(det);
        }

        json result = {
            {"detections",    detections},
            {"count",         (int)objs.size()},
            {"inference_ms",  inference_ms},
            {"image_width",   image.cols},
            {"image_height",  image.rows}
        };

        res.set_content(result.dump(), "application/json");
    });

    printf("YOLOv8 API server starting on %s:%d\n", host.c_str(), port);
    printf("Endpoints:\n");
    printf("  GET  /health  - Health check\n");
    printf("  POST /detect  - Object detection (multipart form with 'image' field)\n");

    if (!svr.listen(host, port)) {
        fprintf(stderr, "Failed to start server on %s:%d\n", host.c_str(), port);
        return 1;
    }

    return 0;
}
