/**
 * Hand-rolled OpenAPI 3.1 spec.
 *
 * We could generate this from Zod/TypeBox schemas, but for an MVP-sized API
 * (5 endpoints) hand-writing keeps it readable, version-controlled, and
 * dependency-free. When we grow past ~15 endpoints, switch to fastify-type-provider.
 */
export const openApiSpec = {
  openapi: "3.1.0",
  info: {
    title: "Augur",
    description: "Adaptive retrieval orchestration layer for RAG systems",
    version: "0.1.0",
    license: { name: "MIT" },
  },
  servers: [{ url: "http://localhost:3001" }],
  paths: {
    "/health": {
      get: {
        summary: "Service health and configuration",
        responses: {
          "200": {
            description: "Service info",
            content: { "application/json": { schema: { $ref: "#/components/schemas/Health" } } },
          },
        },
      },
    },
    "/index": {
      post: {
        summary: "Index documents",
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["documents"],
                properties: {
                  documents: {
                    type: "array",
                    items: { $ref: "#/components/schemas/Document" },
                  },
                },
              },
            },
          },
        },
        responses: {
          "200": {
            description: "Index result",
            content: { "application/json": { schema: { $ref: "#/components/schemas/IndexResponse" } } },
          },
        },
      },
    },
    "/search": {
      post: {
        summary: "Search the index",
        requestBody: {
          required: true,
          content: {
            "application/json": { schema: { $ref: "#/components/schemas/SearchRequest" } },
          },
        },
        responses: {
          "200": {
            description: "Search results + trace",
            content: { "application/json": { schema: { $ref: "#/components/schemas/SearchResponse" } } },
          },
        },
      },
    },
    "/traces": {
      get: {
        summary: "List recent traces",
        parameters: [
          { name: "limit", in: "query", schema: { type: "integer", default: 100 } },
        ],
        responses: { "200": { description: "Traces" } },
      },
      delete: {
        summary: "Clear all traces",
        responses: { "200": { description: "OK" } },
      },
    },
    "/traces/{id}": {
      get: {
        summary: "Get a single trace",
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "string" } },
        ],
        responses: {
          "200": { description: "Trace" },
          "404": { description: "Not found" },
        },
      },
    },
    "/admin/stats": {
      get: { summary: "Adapter and store sizes", responses: { "200": { description: "Stats" } } },
    },
    "/admin/clear": {
      post: { summary: "Clear all indexed data", responses: { "200": { description: "OK" } } },
    },
  },
  components: {
    schemas: {
      Document: {
        type: "object",
        required: ["id", "content"],
        properties: {
          id: { type: "string" },
          content: { type: "string" },
          metadata: { type: "object", additionalProperties: true },
        },
      },
      SearchRequest: {
        type: "object",
        required: ["query"],
        properties: {
          query: { type: "string" },
          documents: {
            type: "array",
            items: { $ref: "#/components/schemas/Document" },
            description: "Optional inline documents for ad-hoc search",
          },
          topK: { type: "integer", default: 10 },
          forceStrategy: { type: "string", enum: ["vector", "keyword", "hybrid", "rerank"] },
          latencyBudgetMs: { type: "integer", description: "Soft latency budget — affects reranking decisions" },
          filter: { type: "object", additionalProperties: true },
        },
      },
      SearchResponse: {
        type: "object",
        properties: {
          results: { type: "array", items: { $ref: "#/components/schemas/SearchResult" } },
          trace: { $ref: "#/components/schemas/SearchTrace" },
        },
      },
      SearchResult: {
        type: "object",
        properties: {
          chunk: { $ref: "#/components/schemas/Chunk" },
          score: { type: "number" },
          rawScores: { type: "object", additionalProperties: { type: "number" } },
        },
      },
      Chunk: {
        type: "object",
        properties: {
          id: { type: "string" },
          documentId: { type: "string" },
          content: { type: "string" },
          index: { type: "integer" },
          metadata: { type: "object", additionalProperties: true },
        },
      },
      SearchTrace: {
        type: "object",
        properties: {
          id: { type: "string" },
          query: { type: "string" },
          startedAt: { type: "string", format: "date-time" },
          totalMs: { type: "number" },
          adapter: { type: "string" },
          embeddingModel: { type: "string" },
          candidates: { type: "integer" },
          decision: {
            type: "object",
            properties: {
              strategy: { type: "string", enum: ["vector", "keyword", "hybrid", "rerank"] },
              reasons: { type: "array", items: { type: "string" } },
              reranked: { type: "boolean" },
              signals: { type: "object", additionalProperties: true },
            },
          },
          spans: { type: "array", items: { $ref: "#/components/schemas/TraceSpan" } },
        },
      },
      TraceSpan: {
        type: "object",
        properties: {
          name: { type: "string" },
          startMs: { type: "number" },
          endMs: { type: "number" },
          durationMs: { type: "number" },
          attributes: { type: "object", additionalProperties: true },
        },
      },
      IndexResponse: {
        type: "object",
        properties: {
          documents: { type: "integer" },
          chunks: { type: "integer" },
          trace: {
            type: "object",
            properties: {
              chunkingMs: { type: "number" },
              embeddingMs: { type: "number" },
              upsertMs: { type: "number" },
              totalMs: { type: "number" },
            },
          },
        },
      },
      Health: {
        type: "object",
        properties: {
          status: { type: "string" },
          adapter: { type: "string" },
          embedder: { type: "string" },
          chunker: { type: "string" },
          router: { type: "string" },
          reranker: { type: "string" },
          capabilities: { type: "object", additionalProperties: { type: "boolean" } },
        },
      },
    },
  },
} as const;
