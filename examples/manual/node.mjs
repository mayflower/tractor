// Node.js with built-in fetch and AbortSignal.timeout; no npm dependencies.
import { readFile } from "node:fs/promises";

const request = await readFile(new URL("./records.json", import.meta.url), "utf8");
const base = (process.env.TRACTOR_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const response = await fetch(`${base}/v1/extract`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: request,
  signal: AbortSignal.timeout(10000),
});
const body = await response.text();
if (!response.ok) {
  throw new Error(`tractor HTTP ${response.status}: ${body}`);
}
// Keeping the raw body preserves decimal number literals during forwarding.
console.log(body);
