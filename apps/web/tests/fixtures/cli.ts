import { generate } from "./generate.ts";

console.log(`Fixtures written to ${generate(Number(process.argv[2]) || 1)}`);
