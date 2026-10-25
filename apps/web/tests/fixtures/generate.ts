// Deterministic test fixtures: distinct RGB PNGs and ImageFolder / YOLO / COCO ZIPs.
// Pure Node (zlib only) so fixtures never depend on private data or network access.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { crc32, deflateSync } from "node:zlib";

// Run from apps/web (npm scripts and Playwright both do).
const OUT = resolve("tests/fixtures/out");

function rng(seed: number) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

function chunk(type: string, data: Buffer) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, crc]);
}

/** An RGB PNG whose pixels are unique per seed (so pixel-group dedupe keeps them apart). */
export function png(width: number, height: number, seed: number): Buffer {
  const random = rng(seed * 2654435761);
  const base = [random() * 255, random() * 255, random() * 255];
  const rows: Buffer[] = [];
  for (let y = 0; y < height; y++) {
    const row = Buffer.alloc(1 + width * 3);
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) row[1 + x * 3 + c] = (base[c]! + random() * 60 + x + y) & 0xff;
    }
    rows.push(row);
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8; // bit depth
  header[9] = 2; // RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(Buffer.concat(rows))),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/** Minimal stored (uncompressed) ZIP writer. */
export function zip(entries: [string, Buffer][]): Buffer {
  const locals: Buffer[] = [];
  const centrals: Buffer[] = [];
  let offset = 0;
  for (const [name, data] of entries) {
    const nameBytes = Buffer.from(name, "utf8");
    const crc = crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x0800, 6); // UTF-8 names
    local.writeUInt16LE(0, 8); // stored
    local.writeUInt16LE(0, 10);
    local.writeUInt16LE(0x21, 12); // 1980-01-01
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBytes.length, 26);
    local.writeUInt16LE(0, 28);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0x0800, 8);
    central.writeUInt16LE(0, 10);
    central.writeUInt16LE(0, 12);
    central.writeUInt16LE(0x21, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(nameBytes.length, 28);
    central.writeUInt32LE(offset, 42);
    locals.push(local, nameBytes, data);
    centrals.push(central, nameBytes);
    offset += local.length + nameBytes.length + data.length;
  }
  const centralBytes = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralBytes.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, centralBytes, end]);
}

function write(name: string, data: Buffer) {
  const path = join(OUT, name);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, data);
}

/** `run` namespaces pixel content so repeated test runs create distinct images. */
export function generate(run = 1) {
  const base = run * 100_000;
  // Loose images for the upload tests; a 200×100 image for coordinate checks.
  for (let i = 0; i < 3; i++) write(`upload/part-${i + 1}.png`, png(48, 32, base + i + 1));
  write("upload/wide-200x100.png", png(200, 100, base + 99));
  write("upload/not-an-image.png", Buffer.from("this is not a png"));

  // ImageFolder with enough distinct images per class for ≥32 test images at 70/15/15.
  const folder: [string, Buffer][] = [];
  for (const [c, name] of ["ok", "defect"].entries()) {
    for (let i = 0; i < 115; i++)
      folder.push([`${name}/img-${i}.png`, png(24, 24, base + 1000 + c * 500 + i)]);
  }
  write("imagefolder.zip", zip(folder));

  // YOLO: centers + sizes; one image deliberately has no label file (a reviewed negative).
  const yolo: [string, Buffer][] = [
    ["data.yaml", Buffer.from("names:\n  0: scratch\n  1: dent\nnc: 2\n")],
  ];
  for (let i = 0; i < 4; i++) {
    yolo.push([`images/train/y-${i}.png`, png(64, 32, base + 3000 + i)]);
    if (i < 3) yolo.push([`labels/train/y-${i}.txt`, Buffer.from(`${i % 2} 0.5 0.5 0.25 0.5\n`)]);
  }
  write("yolo.zip", zip(yolo));

  // COCO: pixel [x, y, w, h] boxes.
  const coco = {
    images: [0, 1, 2].map((i) => ({ id: i + 1, file_name: `c-${i}.png`, width: 100, height: 50 })),
    categories: [
      { id: 7, name: "scratch" },
      { id: 9, name: "pit" },
    ],
    annotations: [
      { id: 1, image_id: 1, category_id: 7, bbox: [10, 5, 40, 20], iscrowd: 0 },
      { id: 2, image_id: 2, category_id: 9, bbox: [50, 10, 30, 30], iscrowd: 0 },
    ],
  };
  write(
    "coco.zip",
    zip([
      ["annotations.json", Buffer.from(JSON.stringify(coco))],
      ...[0, 1, 2].map((i): [string, Buffer] => [
        `images/c-${i}.png`,
        png(100, 50, base + 4000 + i),
      ]),
    ]),
  );
  return OUT;
}
