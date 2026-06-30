using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.UseUrls(Environment.GetEnvironmentVariable("AZURE_EMBEDDED_ASR_URL") ?? "http://127.0.0.1:8791");

builder.Services.AddSingleton<ModelRegistry>();
builder.Services.AddSingleton<SessionManager>();

var app = builder.Build();
app.UseWebSockets(new WebSocketOptions { KeepAliveInterval = TimeSpan.FromSeconds(20) });

app.MapGet("/health", (ModelRegistry registry) => Results.Json(new
{
    status = "ok",
    models = registry.Models.Select(model => new { model.Id, model.Locale, model.ModelDir, model.RecognitionModelName, model.Loaded })
}));

app.MapPost("/models/load", async (ModelRegistry registry) =>
{
    var sw = Stopwatch.StartNew();
    await registry.LoadAllAsync();
    return Results.Json(new { status = "ready", elapsedMs = sw.ElapsedMilliseconds, models = registry.Models.Select(model => model.Id) });
});

app.Map("/asr", async (HttpContext context, SessionManager sessions) =>
{
    if (!context.WebSockets.IsWebSocketRequest)
    {
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
        return;
    }

    using var socket = await context.WebSockets.AcceptWebSocketAsync();
    await sessions.RunAsync(socket, context.RequestAborted);
});

app.Run();

public sealed record ModelInfo(string Id, string Locale, string ModelDir, string LicenseKey, string? RecognitionModelName = null, bool Loaded = false);

public sealed class ModelRegistry
{
    private readonly ConcurrentDictionary<string, ModelInfo> _models = new(StringComparer.OrdinalIgnoreCase);

    public ModelRegistry()
    {
        var configuredRoot = Environment.GetEnvironmentVariable("VOICE_AGENT_AZURE_EMBEDDED_MODEL_ROOT") ?? Path.Combine("models", "azure-embedded");
        var root = ResolveModelRoot(configuredRoot);
        var key = Environment.GetEnvironmentVariable("PASCO_MODEL_KEY") ?? string.Empty;
        Register(new ModelInfo(
            "azure-embedded-zh-CN-35M",
            "zh-CN",
            ResolveModelDir(
                Environment.GetEnvironmentVariable("VOICE_AGENT_AZURE_EMBEDDED_ASR_ZH_CN_MODEL_DIR"),
                Path.Combine(root, "asr", "zh-CN", "encrypted", "35M")),
            key));
        Register(new ModelInfo(
            "azure-embedded-en-GB-35M",
            "en-GB",
            ResolveModelDir(
                Environment.GetEnvironmentVariable("VOICE_AGENT_AZURE_EMBEDDED_ASR_EN_GB_MODEL_DIR"),
                Path.Combine(root, "asr", "en-GB", "encrypted", "v6", "35M"),
                Path.Combine(root, "asr", "en-GB", "encrypted", "35M")),
            key));
    }

    public IEnumerable<ModelInfo> Models => _models.Values.OrderBy(model => model.Id);

    public ModelInfo Resolve(string? idOrLocale)
    {
        if (string.IsNullOrWhiteSpace(idOrLocale)) return _models["azure-embedded-zh-CN-35M"];
        if (_models.TryGetValue(idOrLocale, out var byId)) return byId;
        var byLocale = _models.Values.FirstOrDefault(model => string.Equals(model.Locale, idOrLocale, StringComparison.OrdinalIgnoreCase));
        return byLocale ?? throw new InvalidOperationException($"Unknown model or locale: {idOrLocale}");
    }

    public Task LoadAllAsync()
    {
        foreach (var model in _models.Values) Load(model);
        return Task.CompletedTask;
    }

    public ModelInfo Load(ModelInfo model)
    {
        ValidateModel(model);
        var config = EmbeddedSpeechConfig.FromPath(model.ModelDir);
        var recognitionModel = config.GetSpeechRecognitionModels()
            .FirstOrDefault(candidate => candidate.Locales.Any(locale => string.Equals(locale, model.Locale, StringComparison.OrdinalIgnoreCase)))
            ?? config.GetSpeechRecognitionModels().FirstOrDefault()
            ?? throw new InvalidOperationException($"No embedded speech recognition models found in {model.ModelDir}");
        var loaded = model with { RecognitionModelName = recognitionModel.Name, Loaded = true };
        _models[loaded.Id] = loaded;
        return loaded;
    }

    private void Register(ModelInfo model) => _models[model.Id] = model;

    private static string ResolveModelRoot(string configuredRoot)
    {
        if (Path.IsPathRooted(configuredRoot)) return configuredRoot;
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            var candidate = Path.GetFullPath(Path.Combine(current.FullName, configuredRoot));
            if (Directory.Exists(candidate)) return candidate;
            current = current.Parent;
        }
        return Path.GetFullPath(configuredRoot);
    }

    private static string ResolveModelDir(string? configuredDir, params string[] candidates)
    {
        if (!string.IsNullOrWhiteSpace(configuredDir)) return ResolveModelRoot(configuredDir);
        return candidates.FirstOrDefault(Directory.Exists) ?? candidates[0];
    }

    private static void ValidateModel(ModelInfo model)
    {
        if (string.IsNullOrWhiteSpace(model.LicenseKey)) throw new InvalidOperationException("PASCO_MODEL_KEY is not configured.");
        foreach (var fileName in new[] { "sr.ini", "model_onnx.config", "tokens.list" })
        {
            var path = Path.Combine(model.ModelDir, fileName);
            if (!File.Exists(path)) throw new FileNotFoundException($"Missing model asset: {path}");
        }
        if (Directory.GetFiles(model.ModelDir, "*.onnx").Length < 4) throw new InvalidOperationException($"Model directory has too few ONNX files: {model.ModelDir}");
    }
}

public sealed class SessionManager
{
    private readonly ModelRegistry _registry;

    public SessionManager(ModelRegistry registry) => _registry = registry;

    public async Task RunAsync(WebSocket socket, CancellationToken cancellationToken)
    {
        var buffer = ArrayPool<byte>.Shared.Rent(64 * 1024);
        var model = _registry.Resolve("zh-CN");
        SpeechStreamSession? recognition = null;

        try
        {
            await SendJsonAsync(socket, new { type = "ready" }, cancellationToken);
            while (!cancellationToken.IsCancellationRequested && socket.State == WebSocketState.Open)
            {
                var result = await socket.ReceiveAsync(buffer, cancellationToken);
                if (result.MessageType == WebSocketMessageType.Close) break;

                if (result.MessageType == WebSocketMessageType.Text)
                {
                    var text = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    var message = JsonSerializer.Deserialize<Dictionary<string, string>>(text) ?? new();
                    if (message.TryGetValue("locale", out var locale)) model = _registry.Resolve(locale);
                    if (message.TryGetValue("model", out var modelId)) model = _registry.Resolve(modelId);
                    if (message.GetValueOrDefault("type") is "start" or "config")
                    {
                        recognition ??= await SpeechStreamSession.StartAsync(socket, _registry.Load(model), cancellationToken);
                    }
                    if (message.GetValueOrDefault("type") == "end")
                    {
                        if (recognition is not null)
                        {
                            await recognition.FinishAsync(cancellationToken);
                        }
                        break;
                    }
                }
                else if (result.MessageType == WebSocketMessageType.Binary)
                {
                    recognition ??= await SpeechStreamSession.StartAsync(socket, _registry.Load(model), cancellationToken);
                    recognition.Write(buffer, result.Count);
                }
            }
        }
        catch (Exception exc) when (socket.State == WebSocketState.Open)
        {
            await SendJsonAsync(socket, new { type = "error", errorType = exc.GetType().Name, message = exc.Message }, CancellationToken.None);
        }
        finally
        {
            recognition?.Dispose();
            ArrayPool<byte>.Shared.Return(buffer);
            if (socket.State == WebSocketState.Open)
            {
                await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None);
            }
        }
    }

    public static async Task SendJsonAsync(WebSocket socket, object payload, CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(payload);
        var bytes = Encoding.UTF8.GetBytes(json);
        await socket.SendAsync(bytes, WebSocketMessageType.Text, true, cancellationToken);
    }
}

public sealed class SpeechStreamSession : IDisposable
{
    private readonly WebSocket _socket;
    private readonly ModelInfo _model;
    private readonly Stopwatch _sw = Stopwatch.StartNew();
    private readonly AudioStreamFormat _format;
    private readonly PushAudioInputStream _pushStream;
    private readonly AudioConfig _audioConfig;
    private readonly SpeechRecognizer _recognizer;
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private readonly TaskCompletionSource _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly List<string> _recognizedParts = new();
    private readonly object _textLock = new();
    private int _bytes;
    private bool _finished;

    private SpeechStreamSession(WebSocket socket, ModelInfo model)
    {
        if (string.IsNullOrWhiteSpace(model.RecognitionModelName)) throw new InvalidOperationException($"Model {model.Id} is not loaded.");
        _socket = socket;
        _model = model;
        _format = AudioStreamFormat.GetWaveFormatPCM(16000, 16, 1);
        _pushStream = AudioInputStream.CreatePushStream(_format);
        _audioConfig = AudioConfig.FromStreamInput(_pushStream);
        var embeddedConfig = EmbeddedSpeechConfig.FromPath(model.ModelDir);
        embeddedConfig.SetSpeechRecognitionModel(model.RecognitionModelName, model.LicenseKey);
        _recognizer = new SpeechRecognizer(embeddedConfig, _audioConfig);
        _recognizer.Recognizing += (_, args) =>
        {
            var text = args.Result.Text?.Trim();
            if (!string.IsNullOrWhiteSpace(text))
            {
                _ = SendAsync(new { type = "partial", model = _model.Id, locale = _model.Locale, text, bytes = _bytes });
            }
        };
        _recognizer.Recognized += (_, args) =>
        {
            var text = args.Result.Text?.Trim();
            if (args.Result.Reason == ResultReason.RecognizedSpeech && !string.IsNullOrWhiteSpace(text))
            {
                lock (_textLock) _recognizedParts.Add(text);
                _ = SendAsync(new { type = "partial", model = _model.Id, locale = _model.Locale, text = CurrentText(), bytes = _bytes, stable = true });
            }
        };
        _recognizer.Canceled += (_, args) =>
        {
            if (args.Reason.ToString() != "EndOfStream")
            {
                _ = SendAsync(new { type = "canceled", model = _model.Id, reason = args.Reason.ToString(), details = args.ErrorDetails });
            }
            _stopped.TrySetResult();
        };
        _recognizer.SessionStopped += (_, _) => _stopped.TrySetResult();
    }

    public static async Task<SpeechStreamSession> StartAsync(WebSocket socket, ModelInfo model, CancellationToken cancellationToken)
    {
        var session = new SpeechStreamSession(socket, model);
        await session._recognizer.StartContinuousRecognitionAsync();
        await session.SendAsync(new
        {
            type = "started",
            model = model.Id,
            locale = model.Locale,
            recognitionModel = model.RecognitionModelName
        }, cancellationToken);
        return session;
    }

    public void Write(byte[] buffer, int count)
    {
        if (_finished) return;
        var chunk = new byte[count];
        Buffer.BlockCopy(buffer, 0, chunk, 0, count);
        _pushStream.Write(chunk, count);
        _bytes += count;
    }

    public async Task FinishAsync(CancellationToken cancellationToken)
    {
        if (_finished) return;
        _finished = true;
        _pushStream.Close();
        try
        {
            await _stopped.Task.WaitAsync(TimeSpan.FromSeconds(8), cancellationToken);
        }
        catch (TimeoutException)
        {
            await _recognizer.StopContinuousRecognitionAsync();
        }
        await SendAsync(new
        {
            type = "final",
            model = _model.Id,
            locale = _model.Locale,
            text = CurrentText(),
            bytes = _bytes,
            asrMs = _sw.ElapsedMilliseconds,
            recognitionModel = _model.RecognitionModelName
        }, cancellationToken);
    }

    private string CurrentText()
    {
        lock (_textLock) return string.Join(" ", _recognizedParts).Trim();
    }

    private Task SendAsync(object payload) => SendAsync(payload, CancellationToken.None);

    private async Task SendAsync(object payload, CancellationToken cancellationToken)
    {
        if (_socket.State != WebSocketState.Open) return;
        await _sendLock.WaitAsync(cancellationToken);
        try
        {
            if (_socket.State == WebSocketState.Open)
            {
                await SessionManager.SendJsonAsync(_socket, payload, cancellationToken);
            }
        }
        finally
        {
            _sendLock.Release();
        }
    }

    public void Dispose()
    {
        _recognizer.Dispose();
        _audioConfig.Dispose();
        _pushStream.Dispose();
        _format.Dispose();
        _sendLock.Dispose();
    }
}
