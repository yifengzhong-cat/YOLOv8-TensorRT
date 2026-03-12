//
// YOLOv8 TensorRT HTTP API Server
//
// Provides REST APIs for object detection on images and video streams
// using YOLOv8 TensorRT engines. Implements the cloud analysis platform
// interface specification.
//
// Dependencies: cpp-httplib, nlohmann/json, OpenCV, TensorRT, CUDA
//
#include "httplib.h"
#include "nlohmann/json.hpp"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;
namespace fs = ghc::filesystem;

// =============================================================================
// Base64 encoding/decoding utilities
// =============================================================================
static const std::string BASE64_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const unsigned char* data, size_t len)
{
    std::string result;
    result.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned int val = (data[i] << 16);
        if (i + 1 < len) val |= (data[i + 1] << 8);
        if (i + 2 < len) val |= data[i + 2];
        result.push_back(BASE64_CHARS[(val >> 18) & 0x3F]);
        result.push_back(BASE64_CHARS[(val >> 12) & 0x3F]);
        result.push_back((i + 1 < len) ? BASE64_CHARS[(val >> 6) & 0x3F] : '=');
        result.push_back((i + 2 < len) ? BASE64_CHARS[val & 0x3F] : '=');
    }
    return result;
}

static std::vector<unsigned char> base64_decode(const std::string& encoded)
{
    std::vector<unsigned char> result;
    std::vector<int>           T(256, -1);
    for (int i = 0; i < 64; i++) T[BASE64_CHARS[i]] = i;

    int val = 0, bits = -8;
    for (unsigned char c : encoded) {
        if (T[c] == -1) continue;
        val = (val << 6) + T[c];
        bits += 6;
        if (bits >= 0) {
            result.push_back((val >> bits) & 0xFF);
            bits -= 8;
        }
    }
    return result;
}

static std::string mat_to_base64_jpeg(const cv::Mat& img)
{
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);
    return base64_encode(buf.data(), buf.size());
}

static cv::Mat base64_to_mat(const std::string& b64)
{
    auto decoded = base64_decode(b64);
    return cv::imdecode(decoded, cv::IMREAD_COLOR);
}

static std::string get_current_time_str()
{
    auto        now = std::chrono::system_clock::now();
    std::time_t t   = std::chrono::system_clock::to_time_t(now);
    std::tm     tm;
    localtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

// =============================================================================
// Model configuration structure
// =============================================================================
struct ModelConfig {
    std::string              name;
    std::string              algCode;
    std::string              algDesc;
    std::string              engineFile;
    std::vector<std::string> labels;
    std::vector<std::string> labelDescriptions;
};

// =============================================================================
// Model instance: wraps YOLOv8 engine with its metadata
// =============================================================================
struct ModelInstance {
    std::unique_ptr<YOLOv8>  engine;
    ModelConfig              config;
    std::mutex               mtx;
    std::vector<std::vector<unsigned int>> colors;

    void generateColors()
    {
        colors.clear();
        const std::vector<std::vector<unsigned int>> DEFAULT_COLORS = {
            {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},
            {119, 172, 48},  {77, 190, 238},  {162, 20, 47},   {76, 76, 76},
            {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
            {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0}};
        for (size_t i = 0; i < config.labels.size(); i++) {
            colors.push_back(DEFAULT_COLORS[i % DEFAULT_COLORS.size()]);
        }
    }

    void infer(const cv::Mat& image, std::vector<Object>& objs,
               float score_thres = 0.25f, float iou_thres = 0.65f,
               int topk = 100)
    {
        std::lock_guard<std::mutex> lock(mtx);
        engine->copy_from_Mat(image);
        engine->infer();
        engine->postprocess(objs, score_thres, iou_thres, topk,
                            static_cast<int>(config.labels.size()));
    }

    cv::Mat drawResults(const cv::Mat& image, const std::vector<Object>& objs)
    {
        cv::Mat res;
        YOLOv8::draw_objects(image, res, objs, config.labels, colors);
        return res;
    }
};

// =============================================================================
// Video task structure
// =============================================================================
struct VideoTask {
    std::string analyseId;
    std::string devCode;
    std::string algCode;
    std::string videoUrl;
    int         formatType = 0;
    int         interval   = 60;
    std::string startTime;
    std::string endTime;
    std::string callbackUrl;

    // Rule/ROI configuration
    json rule;

    // Task control
    std::atomic<int>  command{1};  // 0=stop, 1=start, 2=delete
    std::thread       workerThread;
    std::atomic<bool> running{false};

    // Output
    std::string osdVideoUrl;
    std::string outputPath;
};

// =============================================================================
// Global state
// =============================================================================
static std::unordered_map<std::string, std::shared_ptr<ModelInstance>> g_models;    // algCode -> model
static std::unordered_map<std::string, std::shared_ptr<VideoTask>>    g_videoTasks; // analyseId -> task
static std::mutex g_tasksMutex;

static std::string g_modelsDir  = "/workspace/models_dir";
static std::string g_outputDir  = "/workspace/output";
static std::string g_host       = "0.0.0.0";
static int         g_port       = 22266;
static float       g_scoreThres = 0.25f;
static float       g_iouThres   = 0.65f;
static int         g_topk       = 100;

static std::string g_callbackHost = "";
static int         g_callbackPort = 0;

static std::atomic<bool> g_keepAliveRunning{false};
static std::thread       g_keepAliveThread;

// =============================================================================
// Load models from configuration
// =============================================================================
static bool loadModelsFromConfig(const std::string& configPath)
{
    std::ifstream ifs(configPath);
    if (!ifs.is_open()) {
        fprintf(stderr, "ERROR: Cannot open config file: %s\n", configPath.c_str());
        return false;
    }

    json config;
    try {
        ifs >> config;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "ERROR: Failed to parse config: %s\n", e.what());
        return false;
    }

    auto& models = config["models"];
    for (auto& m : models) {
        ModelConfig mc;
        mc.name       = m["name"].get<std::string>();
        mc.algCode    = m["algCode"].get<std::string>();
        mc.algDesc    = m["algDesc"].get<std::string>();
        mc.engineFile = m["engineFile"].get<std::string>();

        // Build labels vector from the map (ordered by key)
        auto& labelMap = m["labels"];
        int   maxIdx   = 0;
        for (auto& [key, val] : labelMap.items()) {
            int idx = std::stoi(key);
            if (idx > maxIdx) maxIdx = idx;
        }
        mc.labels.resize(maxIdx + 1);
        for (auto& [key, val] : labelMap.items()) {
            mc.labels[std::stoi(key)] = val.get<std::string>();
        }

        if (m.contains("labelDescriptions")) {
            auto& descMap = m["labelDescriptions"];
            mc.labelDescriptions.resize(maxIdx + 1);
            for (auto& [key, val] : descMap.items()) {
                int idx = std::stoi(key);
                if (idx < static_cast<int>(mc.labelDescriptions.size()))
                    mc.labelDescriptions[idx] = val.get<std::string>();
            }
        }

        std::string enginePath = g_modelsDir + "/" + mc.engineFile;
        if (!fs::exists(enginePath)) {
            printf("WARNING: Engine file not found: %s (algCode=%s), skipping\n",
                   enginePath.c_str(), mc.algCode.c_str());
            continue;
        }

        printf("Loading model: %s (algCode=%s)\n", mc.name.c_str(), mc.algCode.c_str());
        auto instance    = std::make_shared<ModelInstance>();
        instance->config = mc;
        instance->engine = std::make_unique<YOLOv8>(enginePath);
        instance->engine->make_pipe(true);
        instance->generateColors();
        g_models[mc.algCode] = instance;
        printf("  Loaded successfully: %zu classes\n", mc.labels.size());
    }

    if (config.contains("inference")) {
        auto& inf  = config["inference"];
        if (inf.contains("scoreThreshold"))
            g_scoreThres = inf["scoreThreshold"].get<float>();
        if (inf.contains("iouThreshold"))
            g_iouThres = inf["iouThreshold"].get<float>();
        if (inf.contains("topk"))
            g_topk = inf["topk"].get<int>();
    }

    printf("Loaded %zu model(s)\n", g_models.size());
    return !g_models.empty();
}

// =============================================================================
// Helper: safely stop a video task and join its thread
// =============================================================================
static void stopVideoTask(std::shared_ptr<VideoTask>& task)
{
    task->command = 0;
    if (task->workerThread.joinable()) {
        // Wait for the worker to finish (it checks command flag each frame)
        task->workerThread.join();
    }
    task->running = false;
}

// =============================================================================
// Video task worker: pulls stream, runs inference, saves MP4
// =============================================================================
static void videoTaskWorker(std::shared_ptr<VideoTask> task)
{
    auto modelIt = g_models.find(task->algCode);
    if (modelIt == g_models.end()) {
        fprintf(stderr, "ERROR: No model for algCode=%s\n", task->algCode.c_str());
        task->running = false;
        return;
    }
    auto model = modelIt->second;

    printf("Video task started: analyseId=%s, url=%s\n",
           task->analyseId.c_str(), task->videoUrl.c_str());

    cv::VideoCapture cap;
    cap.open(task->videoUrl);
    if (!cap.isOpened()) {
        fprintf(stderr, "ERROR: Cannot open video stream: %s\n", task->videoUrl.c_str());
        task->running = false;
        return;
    }

    int    fw  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    fh  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;

    // Output MP4 file
    std::string outputFile = g_outputDir + "/" + task->analyseId + ".mp4";
    task->outputPath = outputFile;
    cv::VideoWriter writer(outputFile,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(fw, fh));

    if (!writer.isOpened()) {
        fprintf(stderr, "ERROR: Cannot open video writer: %s\n", outputFile.c_str());
        task->running = false;
        return;
    }

    int    frameCount    = 0;
    int    intervalFrame = static_cast<int>(fps * task->interval);
    if (intervalFrame < 1) intervalFrame = 1;

    cv::Mat             frame, res;
    std::vector<Object> objs;

    auto lastCallback = std::chrono::steady_clock::now();

    while (task->command == 1 && task->running) {
        if (!cap.read(frame)) {
            printf("Video stream ended: analyseId=%s\n", task->analyseId.c_str());
            break;
        }
        if (frame.empty()) continue;

        objs.clear();
        model->infer(frame, objs, g_scoreThres, g_iouThres, g_topk);
        res = model->drawResults(frame, objs);
        writer.write(res);
        frameCount++;

        // Send callback results at interval
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           now - lastCallback).count();

        if (elapsed >= task->interval && !task->callbackUrl.empty() && !objs.empty()) {
            lastCallback = now;

            // Build result detail
            json resultItems = json::array();
            for (auto& obj : objs) {
                resultItems.push_back({
                    {"score",        obj.prob * 100.0f},
                    {"leftTopX",     static_cast<int>(obj.rect.x)},
                    {"leftTopY",     static_cast<int>(obj.rect.y)},
                    {"rightBottomX", static_cast<int>(obj.rect.x + obj.rect.width)},
                    {"rightBottomY", static_cast<int>(obj.rect.y + obj.rect.height)}
                });
            }

            // Build analyse results descriptions
            json analyseResults = json::array();
            for (auto& obj : objs) {
                if (obj.label < static_cast<int>(model->config.labelDescriptions.size())) {
                    analyseResults.push_back(model->config.labelDescriptions[obj.label]);
                }
                else {
                    analyseResults.push_back(model->config.labels[obj.label]);
                }
            }

            json callbackBody = {
                {"algCode",        task->algCode},
                {"analyseId",      task->analyseId},
                {"analyseTime",    get_current_time_str()},
                {"analyseResults", analyseResults},
                {"osdImageName",   task->analyseId + ".jpg"},
                {"osdImageData",   mat_to_base64_jpeg(res)},
                {"resultDetail",   {{
                    {"algCode",    task->algCode},
                    {"resultDesc", "detection"},
                    {"num",        static_cast<int>(objs.size())},
                    {"resultItems", resultItems}
                }}}
            };

            // Send callback (best effort, non-blocking)
            try {
                httplib::Client cli(task->callbackUrl);
                cli.set_connection_timeout(5);
                cli.set_read_timeout(5);
                cli.Post("/analysis/api/v1/analyseResult",
                         callbackBody.dump(), "application/json");
            }
            catch (...) {
                // Callback failure is non-fatal
            }
        }
    }

    writer.release();
    cap.release();
    task->running = false;
    printf("Video task stopped: analyseId=%s, frames=%d, output=%s\n",
           task->analyseId.c_str(), frameCount, outputFile.c_str());
}

// =============================================================================
// Keep-alive heartbeat thread
// =============================================================================
static void keepAliveWorker(const std::string& platformUrl, int port)
{
    while (g_keepAliveRunning) {
        try {
            httplib::Client cli(platformUrl);
            cli.set_connection_timeout(5);
            cli.set_read_timeout(5);
            json body = {
                {"devIP",   g_host},
                {"devPort", port}
            };
            cli.Post("/analysis/api/v1/keepAlive",
                     body.dump(), "application/json");
        }
        catch (...) {
            // Keep-alive failure is non-fatal
        }
        // Sleep 30 seconds between heartbeats
        for (int i = 0; i < 30 && g_keepAliveRunning; i++) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

// =============================================================================
// Helper: create standard JSON response
// =============================================================================
static json makeResponse(const std::string& code,
                         const json& value = nullptr,
                         const std::string& hint = "")
{
    json resp;
    resp["resultCode"]  = code;
    resp["resultValue"] = value;
    resp["resultHint"]  = hint.empty() ? nullptr : json(hint);
    return resp;
}

// =============================================================================
// Command-line argument parsing
// =============================================================================
static void printUsage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "Options:\n"
            "  --config <path>        Models config JSON (default: models_config.json)\n"
            "  --models-dir <path>    Directory with .engine files (default: /workspace/models_dir)\n"
            "  --output-dir <path>    Output directory (default: /workspace/output)\n"
            "  --host <host>          Listen host (default: 0.0.0.0)\n"
            "  --port <port>          Listen port (default: 22266)\n"
            "  --callback-url <url>   Platform callback URL for results\n"
            "  --help                 Show this help\n",
            prog);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv)
{
    std::string configPath = "models_config.json";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
            configPath = argv[++i];
        else if (arg == "--models-dir" && i + 1 < argc)
            g_modelsDir = argv[++i];
        else if (arg == "--output-dir" && i + 1 < argc)
            g_outputDir = argv[++i];
        else if (arg == "--host" && i + 1 < argc)
            g_host = argv[++i];
        else if (arg == "--port" && i + 1 < argc)
            g_port = std::atoi(argv[++i]);
        else if (arg == "--callback-url" && i + 1 < argc)
            g_callbackHost = argv[++i];
        else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Initialize CUDA
    cudaSetDevice(0);

    // Create output directory
    fs::create_directories(g_outputDir);

    // Load models
    if (!loadModelsFromConfig(configPath)) {
        fprintf(stderr, "WARNING: No models loaded. Server will start but inference unavailable.\n");
    }

    httplib::Server svr;

    // =========================================================================
    // GET /health - Health check
    // =========================================================================
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json j = {{"status", "ok"}, {"models_loaded", g_models.size()}};
        res.set_content(j.dump(), "application/json");
    });

    // =========================================================================
    // POST /v1/service/abilities - Algorithm capabilities (4.1.7.1)
    // =========================================================================
    svr.Post("/v1/service/abilities", [](const httplib::Request&, httplib::Response& res) {
        json abilities = json::array();
        for (auto& [code, model] : g_models) {
            json ability;
            ability["algCode"] = model->config.algCode;
            ability["algDesc"] = model->config.algDesc;
            ability["algParams"] = json::array({
                {{"key", "--sensitivity"}, {"value", "灵敏度,范围[1,5]"}}
            });
            abilities.push_back(ability);
        }

        json resp = makeResponse("200", {
            {"abilityInfo", {
                {"number", static_cast<int>(abilities.size())},
                {"ability", abilities}
            }}
        });
        res.set_content(resp.dump(), "application/json");
    });

    // =========================================================================
    // POST /v1/service/videoTask - Video task management (4.1.7.2)
    // =========================================================================
    svr.Post("/v1/service/videoTask", [](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        }
        catch (...) {
            res.status = 400;
            res.set_content(makeResponse("400", nullptr, "消息格式错误").dump(),
                            "application/json");
            return;
        }

        std::string algCode = body.value("algCode", "");
        int command = body.value("command", 1);
        int interval = body.value("interval", 60);
        std::string startTime = body.value("startTime", "");
        std::string endTime = body.value("endTime", "");

        if (g_models.find(algCode) == g_models.end()) {
            res.status = 404;
            res.set_content(
                makeResponse("404", nullptr, "算法编码不存在: " + algCode).dump(),
                "application/json");
            return;
        }

        auto& videoInfoArr = body["videoInfo"];
        if (!videoInfoArr.is_array() || videoInfoArr.empty()) {
            res.status = 400;
            res.set_content(makeResponse("400", nullptr, "videoInfo不能为空").dump(),
                            "application/json");
            return;
        }

        json resultValue = json::array();
        std::lock_guard<std::mutex> lock(g_tasksMutex);

        for (auto& vi : videoInfoArr) {
            std::string analyseId  = vi.value("analyseId", "");
            std::string devCode    = vi.value("devCode", "");
            std::string videoUrl   = vi.value("videoUrl", "");
            int         formatType = vi.value("formatType", 0);

            if (analyseId.empty() || videoUrl.empty()) continue;

            // Stop existing task if any
            if (g_videoTasks.count(analyseId)) {
                auto& existing = g_videoTasks[analyseId];
                stopVideoTask(existing);
                g_videoTasks.erase(analyseId);
            }

            if (command == 1) {
                auto task = std::make_shared<VideoTask>();
                task->analyseId  = analyseId;
                task->devCode    = devCode;
                task->algCode    = algCode;
                task->videoUrl   = videoUrl;
                task->formatType = formatType;
                task->interval   = interval;
                task->startTime  = startTime;
                task->endTime    = endTime;
                if (body.contains("rule"))
                    task->rule = body["rule"];
                task->callbackUrl = g_callbackHost;
                task->command  = 1;
                task->running  = true;

                std::string osdUrl = "http://" + g_host + ":" +
                                     std::to_string(g_port) + "/output/" +
                                     analyseId + ".mp4";
                task->osdVideoUrl = osdUrl;

                g_videoTasks[analyseId] = task;
                task->workerThread = std::thread(videoTaskWorker, task);

                resultValue.push_back({
                    {"analyseId",   analyseId},
                    {"devCode",     devCode},
                    {"osdVideoUrl", osdUrl}
                });
            }
        }

        res.set_content(makeResponse("200", resultValue).dump(), "application/json");
    });

    // =========================================================================
    // POST /v1/service/controlTask - Task control (4.1.7.3)
    // =========================================================================
    svr.Post("/v1/service/controlTask", [](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        }
        catch (...) {
            res.status = 400;
            res.set_content(makeResponse("400", nullptr, "消息格式错误").dump(),
                            "application/json");
            return;
        }

        std::string analyseId = body.value("analyseId", "");
        int command = body.value("command", 0);

        std::lock_guard<std::mutex> lock(g_tasksMutex);
        auto it = g_videoTasks.find(analyseId);
        if (it == g_videoTasks.end()) {
            res.status = 404;
            res.set_content(makeResponse("404", nullptr, "任务不存在").dump(),
                            "application/json");
            return;
        }

        auto& task = it->second;
        switch (command) {
            case 0: // Stop
                stopVideoTask(task);
                res.set_content(
                    makeResponse("200", nullptr, "任务已停止").dump(),
                    "application/json");
                break;
            case 1: // Start/Resume
                if (!task->running) {
                    task->command = 1;
                    task->running = true;
                    task->workerThread = std::thread(videoTaskWorker, task);
                }
                res.set_content(
                    makeResponse("200", nullptr, "任务已启动").dump(),
                    "application/json");
                break;
            case 2: // Delete
                stopVideoTask(task);
                g_videoTasks.erase(it);
                res.set_content(
                    makeResponse("200", nullptr, "任务已删除").dump(),
                    "application/json");
                break;
            default:
                res.status = 400;
                res.set_content(
                    makeResponse("400", nullptr, "无效的command值").dump(),
                    "application/json");
                break;
        }
    });

    // =========================================================================
    // POST /v1/service/imageTask - Image analysis (4.1.7.5)
    // =========================================================================
    svr.Post("/v1/service/imageTask", [](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        }
        catch (...) {
            res.status = 400;
            res.set_content(makeResponse("400", nullptr, "消息格式错误").dump(),
                            "application/json");
            return;
        }

        std::string analyseId = body.value("analyseId", "");
        std::string algCode   = body.value("algCode", "");
        std::string imageData = body.value("imageData", "");

        if (analyseId.empty() || algCode.empty() || imageData.empty()) {
            res.status = 400;
            res.set_content(
                makeResponse("400", nullptr, "缺少必填参数").dump(),
                "application/json");
            return;
        }

        auto it = g_models.find(algCode);
        if (it == g_models.end()) {
            res.status = 404;
            res.set_content(
                makeResponse("404", nullptr, "算法编码不存在: " + algCode).dump(),
                "application/json");
            return;
        }
        auto& model = it->second;

        // Decode base64 image
        cv::Mat image = base64_to_mat(imageData);
        if (image.empty()) {
            res.status = 400;
            res.set_content(
                makeResponse("400", nullptr, "图片解码失败").dump(),
                "application/json");
            return;
        }

        // Run inference
        std::vector<Object> objs;
        model->infer(image, objs, g_scoreThres, g_iouThres, g_topk);

        // Draw results on image
        cv::Mat osdImage = model->drawResults(image, objs);

        // Build result detail
        json resultItems = json::array();
        for (auto& obj : objs) {
            resultItems.push_back({
                {"score",        obj.prob * 100.0f},
                {"leftTopX",     static_cast<int>(obj.rect.x)},
                {"leftTopY",     static_cast<int>(obj.rect.y)},
                {"rightBottomX", static_cast<int>(obj.rect.x + obj.rect.width)},
                {"rightBottomY", static_cast<int>(obj.rect.y + obj.rect.height)}
            });
        }

        // Build analyse results
        json analyseResults = json::array();
        for (auto& obj : objs) {
            if (obj.label < static_cast<int>(model->config.labelDescriptions.size()) &&
                !model->config.labelDescriptions[obj.label].empty()) {
                analyseResults.push_back(model->config.labelDescriptions[obj.label]);
            }
            else if (obj.label < static_cast<int>(model->config.labels.size())) {
                analyseResults.push_back(model->config.labels[obj.label]);
            }
        }

        json resultValue = {
            {"analyseTime",    get_current_time_str()},
            {"analyseResults", analyseResults},
            {"rawImageName",   analyseId + "_raw.jpg"},
            {"rawImageData",   mat_to_base64_jpeg(image)},
            {"osdImageName",   analyseId + "_osd.jpg"},
            {"osdImageData",   mat_to_base64_jpeg(osdImage)},
            {"resultDetail",   {{
                {"algCode",    algCode},
                {"resultDesc", model->config.algDesc},
                {"num",        static_cast<int>(objs.size())},
                {"resultItems", resultItems}
            }}}
        };

        res.set_content(makeResponse("200", resultValue).dump(), "application/json");
    });

    // =========================================================================
    // POST /v1/service/uploadSamples - Sample data upload (4.1.7.6)
    // =========================================================================
    svr.Post("/v1/service/uploadSamples", [](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        }
        catch (...) {
            res.status = 400;
            res.set_content(makeResponse("400", nullptr, "消息格式错误").dump(),
                            "application/json");
            return;
        }

        std::string fileId = body.value("fileId", "");
        std::string md5    = body.value("md5", "");

        if (fileId.empty() || md5.empty()) {
            res.status = 400;
            res.set_content(
                makeResponse("400", nullptr, "缺少必填参数").dump(),
                "application/json");
            return;
        }

        json resp = makeResponse("200",
            {{"fileId", fileId}},
            "接收样本成功");
        res.set_content(resp.dump(), "application/json");
    });

    // =========================================================================
    // POST /analysis/api/v1/updateAnalyseID - Update analysis ID (4.1.7.8)
    // =========================================================================
    svr.Post("/analysis/api/v1/updateAnalyseID", [](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        }
        catch (...) {
            res.status = 400;
            res.set_content(makeResponse("400", nullptr, "消息格式错误").dump(),
                            "application/json");
            return;
        }

        std::string oldId = body.value("oldAnalyseId", "");
        std::string newId = body.value("newAnalyseId", "");

        if (oldId.empty() || newId.empty()) {
            res.status = 400;
            res.set_content(
                makeResponse("400", nullptr, "缺少必填参数").dump(),
                "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(g_tasksMutex);
        auto it = g_videoTasks.find(oldId);
        if (it == g_videoTasks.end()) {
            res.status = 404;
            res.set_content(
                makeResponse("404", nullptr, "原分析ID不存在").dump(),
                "application/json");
            return;
        }

        auto task       = it->second;
        task->analyseId = newId;

        // Update the video URL if it contains the old analyseId
        std::string& url = task->videoUrl;
        size_t pos = url.find(oldId);
        if (pos != std::string::npos) {
            url.replace(pos, oldId.length(), newId);
        }

        g_videoTasks.erase(it);
        g_videoTasks[newId] = task;

        // If task was running, restart with new URL
        if (task->running) {
            stopVideoTask(task);
            task->command = 1;
            task->running = true;
            task->workerThread = std::thread(videoTaskWorker, task);
        }

        json resp = makeResponse("200",
            {{"newAnalyseId", newId}},
            "更新成功");
        res.set_content(resp.dump(), "application/json");
    });

    // =========================================================================
    // Static file serving for output directory (for OSD video access)
    // =========================================================================
    svr.set_mount_point("/output", g_outputDir);

    // =========================================================================
    // Start keep-alive thread if callback URL is configured
    // =========================================================================
    if (!g_callbackHost.empty()) {
        g_keepAliveRunning = true;
        g_keepAliveThread  = std::thread(keepAliveWorker, g_callbackHost, g_port);
    }

    // =========================================================================
    // Start server
    // =========================================================================
    printf("\n============================================\n");
    printf("  YOLOv8-TensorRT API Server\n");
    printf("============================================\n");
    printf("Host:        %s\n", g_host.c_str());
    printf("Port:        %d\n", g_port);
    printf("Models:      %zu loaded\n", g_models.size());
    printf("Output dir:  %s\n", g_outputDir.c_str());
    printf("\nEndpoints:\n");
    printf("  GET  /health                          - Health check\n");
    printf("  POST /v1/service/abilities            - Algorithm capabilities\n");
    printf("  POST /v1/service/videoTask             - Video task management\n");
    printf("  POST /v1/service/controlTask           - Task control\n");
    printf("  POST /v1/service/imageTask             - Image analysis\n");
    printf("  POST /v1/service/uploadSamples         - Sample upload\n");
    printf("  POST /analysis/api/v1/updateAnalyseID  - Update analysis ID\n");
    printf("  GET  /output/*                         - Access output files\n");
    printf("\n");

    if (!svr.listen(g_host, g_port)) {
        fprintf(stderr, "Failed to start server on %s:%d\n", g_host.c_str(), g_port);
        return 1;
    }

    // Cleanup
    g_keepAliveRunning = false;
    if (g_keepAliveThread.joinable())
        g_keepAliveThread.join();

    return 0;
}
