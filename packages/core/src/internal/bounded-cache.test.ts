import test from "node:test";
import assert from "node:assert/strict";
import { BoundedCache } from "./bounded-cache.js";

test("BoundedCache: rejects invalid capacity", () => {
  assert.throws(() => new BoundedCache(0));
  assert.throws(() => new BoundedCache(-1));
  assert.throws(() => new BoundedCache(1.5));
});

test("BoundedCache: get returns set value", () => {
  const c = new BoundedCache<string, number>(2);
  c.set("a", 1);
  assert.equal(c.get("a"), 1);
  assert.equal(c.size(), 1);
});

test("BoundedCache: get on missing key returns undefined", () => {
  const c = new BoundedCache<string, number>(2);
  assert.equal(c.get("nope"), undefined);
});

test("BoundedCache: evicts oldest when capacity exceeded", () => {
  const c = new BoundedCache<string, number>(2);
  c.set("a", 1);
  c.set("b", 2);
  c.set("c", 3); // evicts a
  assert.equal(c.get("a"), undefined);
  assert.equal(c.get("b"), 2);
  assert.equal(c.get("c"), 3);
  assert.equal(c.size(), 2);
});

test("BoundedCache: get touches the entry (LRU semantics)", () => {
  const c = new BoundedCache<string, number>(2);
  c.set("a", 1);
  c.set("b", 2);
  c.get("a"); // a is now MRU
  c.set("c", 3); // evicts b, not a
  assert.equal(c.get("a"), 1);
  assert.equal(c.get("b"), undefined);
  assert.equal(c.get("c"), 3);
});

test("BoundedCache: set on existing key moves to MRU", () => {
  const c = new BoundedCache<string, number>(2);
  c.set("a", 1);
  c.set("b", 2);
  c.set("a", 10); // updates a, makes it MRU
  c.set("c", 3); // evicts b
  assert.equal(c.get("a"), 10);
  assert.equal(c.get("b"), undefined);
});

test("BoundedCache: clear empties the store", () => {
  const c = new BoundedCache<string, number>(2);
  c.set("a", 1);
  c.set("b", 2);
  c.clear();
  assert.equal(c.size(), 0);
  assert.equal(c.get("a"), undefined);
});

test("BoundedCache: delete removes key", () => {
  const c = new BoundedCache<string, number>(2);
  c.set("a", 1);
  assert.equal(c.delete("a"), true);
  assert.equal(c.delete("a"), false);
  assert.equal(c.size(), 0);
});

test("BoundedCache: has returns presence", () => {
  const c = new BoundedCache<string, number>(2);
  c.set("a", 1);
  assert.equal(c.has("a"), true);
  assert.equal(c.has("b"), false);
});
