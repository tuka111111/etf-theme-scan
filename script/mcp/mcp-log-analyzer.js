// scripts/mcp/mcp-log-analyzer.js
import { Server } from "@modelcontextprotocol/sdk/server";
import fs from "fs";

const server = new Server({
  name: "log-analyzer",
  version: "1.1.0"
});

const SUPPORTED_SCHEMA_VERSIONS = [1, 2];

function readJsonLines(path) {
  const lines = fs.readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line, idx) => {
      try {
        return JSON.parse(line);
      } catch (e) {
        throw new Error(`Invalid JSON at line ${idx + 1}`);
      }
    });

  return lines.map((row, idx) => normalizeRow(row, idx));
}

function normalizeRow(row, idx) {
  if (!row.schema_version && row.schema_version !== 0) {
    throw new Error(`Missing schema_version at line ${idx + 1}`);
  }
  if (!SUPPORTED_SCHEMA_VERSIONS.includes(row.schema_version)) {
    throw new Error(
      `Unsupported schema_version=${row.schema_version} at line ${idx + 1}`
    );
  }
  switch (row.schema_version) {
    case 1:
      return normalizeV1(row);
    case 2:
      return normalizeV2(row);
    default:
      throw new Error(
        `Unsupported schema_version=${row.schema_version} at line ${idx + 1}`
      );
  }
}

function normalizeV1(row) {
  return {
    schema_version: "canonical",
    ts: requireString(row.ts, "ts"),
    decision_id: toStringOrNull(row.decision_id, "decision_id"),
    status: requireStatus(row.status),
    env: toStringOrNull(row.env, "env"),
    reason_codes: toStringArray(row.reason_codes, "reason_codes"),
    raw_schema_version: 1
  };
}

function normalizeV2(row) {
  return {
    schema_version: "canonical",
    ts: requireString(row.ts, "ts"),
    decision_id: toStringOrNull(row.decision_id, "decision_id"),
    status: requireStatus(row.status),
    env: toStringOrNull(row.env, "env"),
    reason_codes: toStringArray(row.reason_codes, "reason_codes"),
    raw_schema_version: 2
  };
}

function requireString(value, field) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Invalid ${field}`);
  }
  return value;
}

function toStringOrNull(value, field) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  if (typeof value !== "string") {
    throw new Error(`Invalid ${field}`);
  }
  return value;
}

function toStringArray(value, field) {
  if (value === null || value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new Error(`Invalid ${field}`);
  }
  return value.map(v => {
    if (typeof v !== "string") {
      throw new Error(`Invalid ${field}`);
    }
    return v;
  });
}

function requireStatus(value) {
  if (value !== "ok" && value !== "error") {
    throw new Error("Invalid status");
  }
  return value;
}

/* ===== MCP tools ===== */

server.tool("schema_info", async () => {
  return {
    supported_versions: SUPPORTED_SCHEMA_VERSIONS,
    canonical_schema: "canonical"
  };
});

server.tool("summary", async ({ path }) => {
  const rows = readJsonLines(path);
  return {
    total: rows.length,
    errors: rows.filter(r => r.status === "error").length
  };
});

server.tool("find_missing_decision_id", async ({ path }) => {
  const rows = readJsonLines(path);
  return rows.filter(r => !r.decision_id);
});

server.tool("grep", async ({ path, key, value }) => {
  const rows = readJsonLines(path);
  return rows.filter(r => r[key] === value);
});

server.start();
