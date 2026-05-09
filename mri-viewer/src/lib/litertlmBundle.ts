import { ByteBuffer } from 'flatbuffers';
import { decompressSync } from 'fflate';

export type LiteRtLmSectionDataType =
  | 'GenericBinaryData'
  | 'Deprecated'
  | 'TFLiteModel'
  | 'SP_Tokenizer'
  | 'LlmMetadataProto'
  | 'HF_Tokenizer_Zlib'
  | 'TFLiteWeights'
  | 'NONE';

export interface LiteRtLmSection {
  dataType: LiteRtLmSectionDataType;
  beginOffset: number;
  endOffset: number;
  metadata: Record<string, string | number | boolean>;
  bytes?: Uint8Array;
}

export interface LiteRtLmBundle {
  version: string;
  sections: LiteRtLmSection[];
  getSectionByModelType(modelType: string): LiteRtLmSection | undefined;
  getHfTokenizerJson(): unknown;
}

const MAGIC = 'LITERTLM';
const HEADER_BEGIN_BYTE_OFFSET = 32;
const HEADER_END_LOCATION_BYTE_OFFSET = 24;

const DATA_TYPE_NAMES: Record<number, LiteRtLmSectionDataType> = {
  0: 'NONE',
  1: 'GenericBinaryData',
  2: 'Deprecated',
  3: 'TFLiteModel',
  4: 'SP_Tokenizer',
  5: 'LlmMetadataProto',
  6: 'HF_Tokenizer_Zlib',
  7: 'TFLiteWeights',
};

const VALUE_TYPE = {
  UInt8: 1,
  Int8: 2,
  UInt16: 3,
  Int16: 4,
  UInt32: 5,
  Int32: 6,
  Float32: 7,
  Bool: 8,
  StringValue: 9,
  UInt64: 10,
  Int64: 11,
  Double: 12,
} as const;

function readAscii(bytes: Uint8Array, start: number, length: number) {
  return String.fromCharCode(...bytes.subarray(start, start + length));
}

function readUint64LE(view: DataView, offset: number) {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`LiteRT-LM offset exceeds safe JavaScript range: ${value.toString()}`);
  }
  return Number(value);
}

function tableField(bb: ByteBuffer, tablePosition: number, vtableOffset: number) {
  return bb.__offset(tablePosition, vtableOffset);
}

function tableString(bb: ByteBuffer, tablePosition: number, vtableOffset: number) {
  const offset = tableField(bb, tablePosition, vtableOffset);
  return offset ? String(bb.__string(tablePosition + offset)) : undefined;
}

function tableU8(bb: ByteBuffer, tablePosition: number, vtableOffset: number) {
  const offset = tableField(bb, tablePosition, vtableOffset);
  return offset ? bb.readUint8(tablePosition + offset) : 0;
}

function tableU64(bb: ByteBuffer, tablePosition: number, vtableOffset: number) {
  const offset = tableField(bb, tablePosition, vtableOffset);
  return offset ? Number(bb.readUint64(tablePosition + offset)) : 0;
}

function tableVector(bb: ByteBuffer, tablePosition: number, vtableOffset: number) {
  const offset = tableField(bb, tablePosition, vtableOffset);
  if (!offset) return { start: 0, length: 0 };
  return {
    start: bb.__vector(tablePosition + offset),
    length: bb.__vector_len(tablePosition + offset),
  };
}

function parseValue(bb: ByteBuffer, valueType: number, unionOffset: number) {
  if (!unionOffset) return undefined;
  const valuePosition = bb.__indirect(unionOffset);
  const valueOffset = tableField(bb, valuePosition, 4);
  if (!valueOffset) return undefined;
  const fieldPosition = valuePosition + valueOffset;

  switch (valueType) {
    case VALUE_TYPE.UInt8:
      return bb.readUint8(fieldPosition);
    case VALUE_TYPE.Int8:
      return bb.readInt8(fieldPosition);
    case VALUE_TYPE.UInt16:
      return bb.readUint16(fieldPosition);
    case VALUE_TYPE.Int16:
      return bb.readInt16(fieldPosition);
    case VALUE_TYPE.UInt32:
      return bb.readUint32(fieldPosition);
    case VALUE_TYPE.Int32:
      return bb.readInt32(fieldPosition);
    case VALUE_TYPE.Float32:
      return bb.readFloat32(fieldPosition);
    case VALUE_TYPE.Bool:
      return bb.readUint8(fieldPosition) !== 0;
    case VALUE_TYPE.StringValue:
      return String(bb.__string(fieldPosition));
    case VALUE_TYPE.UInt64:
      return Number(bb.readUint64(fieldPosition));
    case VALUE_TYPE.Int64:
      return Number(bb.readInt64(fieldPosition));
    case VALUE_TYPE.Double:
      return bb.readFloat64(fieldPosition);
    default:
      return undefined;
  }
}

function parseKeyValuePair(bb: ByteBuffer, keyValuePosition: number) {
  const key = tableString(bb, keyValuePosition, 4);
  const valueType = tableU8(bb, keyValuePosition, 6);
  const valueOffset = tableField(bb, keyValuePosition, 8);
  const value = parseValue(bb, valueType, keyValuePosition + valueOffset);

  if (!key || value === undefined) return undefined;
  return [key, value] as const;
}

function parseSectionObject(
  bb: ByteBuffer,
  sectionPosition: number,
  fileSize: number,
  fileBytes?: Uint8Array
): LiteRtLmSection {
  const items = tableVector(bb, sectionPosition, 4);
  const metadata: Record<string, string | number | boolean> = {};

  for (let index = 0; index < items.length; index += 1) {
    const keyValuePosition = bb.__indirect(items.start + index * 4);
    const entry = parseKeyValuePair(bb, keyValuePosition);
    if (entry) {
      metadata[entry[0]] = entry[1];
    }
  }

  const beginOffset = tableU64(bb, sectionPosition, 6);
  const endOffset = tableU64(bb, sectionPosition, 8);
  const dataTypeId = tableU8(bb, sectionPosition, 10);
  const dataType = DATA_TYPE_NAMES[dataTypeId] ?? 'NONE';

  if (beginOffset < 0 || endOffset > fileSize || endOffset < beginOffset) {
    throw new Error(`Invalid LiteRT-LM section offset range: ${beginOffset}-${endOffset}`);
  }

  return {
    dataType,
    beginOffset,
    endOffset,
    metadata,
    bytes: fileBytes?.subarray(beginOffset, endOffset),
  };
}

export function parseLiteRtLmBundle(fileBytes: Uint8Array): LiteRtLmBundle {
  return parseLiteRtLmBundleHeader(fileBytes.subarray(0, 16 * 1024), fileBytes.byteLength, fileBytes);
}

export function decodeHfTokenizerJson(sectionBytes: Uint8Array): unknown {
  const tokenizerView = new DataView(sectionBytes.buffer, sectionBytes.byteOffset, sectionBytes.byteLength);
  const expectedSize = readUint64LE(tokenizerView, 0);
  const decompressed = decompressSync(sectionBytes.subarray(8));
  if (decompressed.byteLength !== expectedSize) {
    throw new Error(`Decompressed tokenizer size mismatch: expected ${expectedSize}, got ${decompressed.byteLength}`);
  }

  return JSON.parse(new TextDecoder().decode(decompressed));
}

export function parseLiteRtLmBundleHeader(headerBlock: Uint8Array, fileSize: number, fileBytes?: Uint8Array): LiteRtLmBundle {
  const magic = readAscii(headerBlock, 0, MAGIC.length);
  if (magic !== MAGIC) {
    throw new Error(`Invalid LiteRT-LM magic header: ${magic}`);
  }

  const view = new DataView(headerBlock.buffer, headerBlock.byteOffset, headerBlock.byteLength);
  const major = view.getUint32(8, true);
  const minor = view.getUint32(12, true);
  const patch = view.getUint32(16, true);
  const headerEndOffset = readUint64LE(view, HEADER_END_LOCATION_BYTE_OFFSET);

  if (headerEndOffset <= HEADER_BEGIN_BYTE_OFFSET || headerEndOffset > headerBlock.byteLength) {
    throw new Error(`Invalid LiteRT-LM header end offset: ${headerEndOffset}`);
  }

  const headerBytes = headerBlock.subarray(HEADER_BEGIN_BYTE_OFFSET, headerEndOffset);
  const bb = new ByteBuffer(headerBytes);
  const rootPosition = bb.readInt32(bb.position()) + bb.position();
  const sectionMetadataOffset = tableField(bb, rootPosition, 6);

  if (!sectionMetadataOffset) {
    throw new Error('LiteRT-LM bundle has no section metadata.');
  }

  const sectionMetadataPosition = bb.__indirect(rootPosition + sectionMetadataOffset);
  const objects = tableVector(bb, sectionMetadataPosition, 4);
  const sections: LiteRtLmSection[] = [];

  for (let index = 0; index < objects.length; index += 1) {
    const sectionPosition = bb.__indirect(objects.start + index * 4);
    sections.push(parseSectionObject(bb, sectionPosition, fileSize, fileBytes));
  }

  return {
    version: `${major}.${minor}.${patch}`,
    sections,
    getSectionByModelType(modelType: string) {
      return sections.find((section) => section.metadata.model_type === modelType);
    },
    getHfTokenizerJson() {
      const section = sections.find((candidate) => candidate.dataType === 'HF_Tokenizer_Zlib');
      if (!section) {
        throw new Error('LiteRT-LM bundle does not include an HF tokenizer section.');
      }
      if (!section.bytes) {
        throw new Error('HF tokenizer section bytes were not loaded.');
      }

      return decodeHfTokenizerJson(section.bytes);
    },
  };
}
