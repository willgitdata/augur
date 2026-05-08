import { test } from "node:test";
import assert from "node:assert/strict";
import { computeSignals } from "./signals.js";

test("signals: wordCount / avgWordLen on typical query", () => {
  const s = computeSignals("how do I tune autovacuum");
  assert.equal(s.wordCount, 5);
  assert.ok(s.avgWordLen > 2);
});

test("signals: hasQuotedPhrase detects double and single quotes", () => {
  assert.equal(computeSignals('look up "exact phrase" please').hasQuotedPhrase, true);
  assert.equal(computeSignals("compare 'borrow checker' rules").hasQuotedPhrase, true);
  assert.equal(computeSignals("no quotes here").hasQuotedPhrase, false);
});

test("signals: hasSpecificTokens for codes and IDs", () => {
  assert.equal(computeSignals("ERR_CONNECTION_REFUSED 4101").hasSpecificTokens, true);
  assert.equal(computeSignals("0xCAFEBABE").hasSpecificTokens, true);
  assert.equal(computeSignals("plain English text").hasSpecificTokens, false);
});

test("signals: hasCodeLike for camelCase identifiers", () => {
  assert.equal(computeSignals("useState in React").hasCodeLike, true);
});

test("signals: hasCodeLike for snake_case identifiers", () => {
  assert.equal(computeSignals("pg_repack production").hasCodeLike, true);
});

test("signals: hasCodeLike for dotted identifiers", () => {
  assert.equal(computeSignals("rbac.authorization.k8s.io").hasCodeLike, true);
});

test("signals: hasCodeLike false on plain prose", () => {
  assert.equal(computeSignals("how should I deploy this").hasCodeLike, false);
});

test("signals: hasDateOrVersion for years", () => {
  assert.equal(computeSignals("released in 2018").hasDateOrVersion, true);
});

test("signals: hasDateOrVersion for semver", () => {
  assert.equal(computeSignals("upgrade to v3.1.2").hasDateOrVersion, true);
  assert.equal(computeSignals("TLS 1.3 handshake").hasDateOrVersion, true);
});

test("signals: hasDateOrVersion for RFC and CVE", () => {
  assert.equal(computeSignals("RFC 7519").hasDateOrVersion, true);
  assert.equal(computeSignals("CVE-2024-3094").hasDateOrVersion, true);
});

test("signals: questionType=procedural for how/why", () => {
  assert.equal(computeSignals("how do I deploy with zero downtime").questionType, "procedural");
  assert.equal(computeSignals("why does my pod restart").questionType, "procedural");
});

test("signals: questionType=definitional for what is", () => {
  assert.equal(computeSignals("what is the GIL").questionType, "definitional");
  assert.equal(computeSignals("what does FSDP do").questionType, "definitional");
});

test("signals: questionType=factoid for who/when/where/which", () => {
  assert.equal(computeSignals("who created PgBouncer").questionType, "factoid");
  assert.equal(computeSignals("when was TLS 1.3 standardized").questionType, "factoid");
  assert.equal(computeSignals("which RBAC object handles permissions").questionType, "factoid");
});

test("signals: questionType=null when not a question", () => {
  assert.equal(computeSignals("deploy production rollout pipeline").questionType, null);
});

test("signals: hasNamedEntity for capitalized mid-query token", () => {
  assert.equal(computeSignals("setup PgBouncer for production").hasNamedEntity, true);
});

test("signals: hasNamedEntity ignores sentence-initial caps", () => {
  // "How" at position 0 is just sentence-start capitalization, not entity.
  assert.equal(computeSignals("How do I deploy").hasNamedEntity, false);
});

test("signals: hasNegation for not / without / vs", () => {
  assert.equal(computeSignals("vacuum without locking writes").hasNegation, true);
  assert.equal(computeSignals("HTTP vs HTTPS").hasNegation, true);
  assert.equal(computeSignals("not restarting pod").hasNegation, true);
});

test("signals: hasNegation false on plain query", () => {
  assert.equal(computeSignals("redis cluster setup").hasNegation, false);
});

test("signals: language ja on Japanese (hiragana/katakana)", () => {
  // "How to delete duplicate rows in PostgreSQL" — uses hiragana の/を/する
  assert.equal(computeSignals("PostgreSQL の重複行を削除する方法").language, "ja");
});

test("signals: language zh on Chinese (Han only)", () => {
  assert.equal(computeSignals("如何配置连接池").language, "zh");
});

test("signals: language ko on Korean (hangul)", () => {
  assert.equal(computeSignals("연결 풀 설정 방법").language, "ko");
});

test("signals: language ru on Cyrillic", () => {
  assert.equal(computeSignals("настройка пула соединений Postgres").language, "ru");
});

test("signals: language ar on Arabic", () => {
  assert.equal(computeSignals("تجميع اتصالات قاعدة البيانات").language, "ar");
});

test("signals: language hi on Devanagari", () => {
  assert.equal(computeSignals("कनेक्शन पूल कॉन्फ़िगर करें").language, "hi");
});

test("signals: language en on Spanish (Latin script)", () => {
  // Spanish uses Latin script; we leave it as "en" since BM25 with a Latin
  // tokenizer handles it passably and Spanish docs are often code-mixed.
  assert.equal(computeSignals("cómo configurar pool de conexiones").language, "en");
});

test("signals: language en on plain English", () => {
  assert.equal(computeSignals("how do I deploy").language, "en");
});
