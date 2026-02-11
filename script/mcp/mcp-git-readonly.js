// mcp-git-readonly.js
import { Server } from "@modelcontextprotocol/sdk/server";
import { execSync } from "child_process";

const server = new Server({
  name: "git-readonly",
  version: "1.0.0"
});

server.tool("status", async () => {
  return { output: execSync("git status --short").toString() };
});

server.tool("diff", async () => {
  return { output: execSync("git diff").toString() };
});

server.tool("log", async ({ limit = 5 }) => {
  return {
    output: execSync(`git log -${limit} --oneline`).toString()
  };
});

server.start();

