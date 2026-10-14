/*
 * stellar-ai 无头自检入口（Node）
 * 运行：node test/selftest.js
 */
const C = require('../src/autograd');
const M = require('../src/models');
const { runAll } = require('./checks');

const { pass, fail } = runAll(C, M, (line) => console.log(line));
process.exit(fail === 0 ? 0 : 1);
