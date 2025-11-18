package ai.mlc.mlcllm

import android.content.Context
import android.os.BatteryManager
import java.io.File
import java.io.OutputStreamWriter
import java.lang.Math.exp
import java.nio.charset.Charset
import ai.mlc.mlcllm.OpenAIProtocol
import android.util.Log
import kotlin.collections.mutableListOf

// CSV escaping
fun escapeCsv(value: String?): String {
    if (value == null) return ""
    val needQuotes = value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r')
    var out = value.replace("\"", "\"\"")
    if (needQuotes) out = "\"$out\""
    return out
}

// memory used by current VM (bytes)
fun runtimeUsedBytes(): Long {
    val rt = Runtime.getRuntime()
    return rt.totalMemory() - rt.freeMemory()
}

// Attempt to read an energy counter if available (device dependent).
// returns energy counter as Long or null if not available.
fun readEnergyCounterMicroWh(context: Context): Long? {
    // NOTE: unit and availability depends on device. Many devices won't provide this.
    val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager ?: return null
    return try {
        // This property often returns nanoWatt-hours or microWatt-hours depending on platform.
        // Treat returned value as microWatt-hours for downstream calculations; but it's device dependent.
        val valLong = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER)
        if (valLong == Long.MIN_VALUE) null else valLong
    } catch (e: Throwable) {
        null
    }
}

// Compute perplexity from a list of token logprobs (log probabilities).
// logProbs are expected to be natural log (ln p). If they are base-10, you'll need to convert.
// If you have per-token log probs, perplexity = exp(- sum(ln p_i) / N)
fun computePerplexityFromLogProbs(logProbs: List<OpenAIProtocol.LogProbsContent>?): Float? {
    if (logProbs == null || logProbs.isEmpty()) return null
    // sum of ln p_i
    var s = 0.0
    for (lp in logProbs) {
        val value = lp.logprob
        s += value
    }
    val avgNegLog = -s / logProbs.size
    val perp = exp(avgNegLog)
    return perp.toFloat()
}

fun appendPromptResultToCsv(context: Context, data: OpenAIProtocol.PromptResultData) {
    val filename = "prompt_results.csv"
    val file = File(context.filesDir, filename)
    val writeHeader = !file.exists() || file.length() == 0L

    // build CSV line
    val values = listOf(
        data.prompt,
        data.model_name,
        data.response,
        data.prompt_tokens.toString(),
        data.completion_tokens.toString(),
        String.format("%.3f", data.prefill_tokens_per_s),
        String.format("%.3f", data.decode_tokens_per_s),
        data.num_prefill_tokens.toString(),
        String.format("%.6f", data.total_latency),
        String.format("%.6f", data.time_to_first_token),
        String.format("%.3f", data.throughput_tps),
        data.energy_consumed_uWh?.let { String.format("%.3f", it) } ?: "",
        data.avg_power_w?.let { String.format("%.6f", it) } ?: "",
        String.format("%.3f", data.cpu_time_ms),
        String.format("%.3f", data.mem_delta_mb),
        data.perplexity_score?.let { String.format("%.6f", it) } ?: "",
        String.format("%.3f", data.start_epoch_ms / 1000.0),
        String.format("%.3f", data.end_epoch_ms / 1000.0),
    )
    val line = values.joinToString(",") { escapeCsv(it) } + "\n"

    OutputStreamWriter(file.outputStream().apply { if (!writeHeader) {} }, Charset.forName("UTF-8")).use { writer ->
        if (writeHeader) {
            val header = listOf(
                "prompt",
                "model_name",
                "response",
                "prompt_tokens",
                "completion_tokens",
                "prefill_tokens_per_s",
                "decode_tokens_per_s",
                "num_prefill_tokens",
                "total_latency_s",
                "time_to_first_token_s",
                "throughput_tps",
                "energy_consumed_uWh",
                "avg_power_w",
                "cpu_time_ms",
                "mem_delta_mb",
                "perplexity_score"
            ).joinToString(",") + "\n"
            writer.write(header)
        }
        writer.write(line)
        writer.flush()
    }
    Log.v("requestGenerate", "Successfully wrote to prompt result CSV: ${file.absolutePath}")
}
