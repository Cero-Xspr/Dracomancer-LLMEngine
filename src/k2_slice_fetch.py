#!/usr/bin/env python3
"""k2_slice_fetch.py — 用正确的绝对偏移 (8+header_len+off) 从 modelscope 拉张量切片。"""
import struct, json, subprocess, os, sys

def fetch(shard_no, outdir='/media/Data-1/npu-direct/hybrid/.st_fix'):
    url_file = f'/tmp/st{shard_no}_final_url.txt'
    head_file = f'/tmp/st{shard_no}_head.bin'
    url = open(url_file).read().strip()
    raw = open(head_file, 'rb').read()
    n, = struct.unpack('<Q', raw[:8])
    hdr = json.loads(raw[8:8+n])
    os.makedirs(outdir, exist_ok=True)
    print(f'shard {shard_no}: header_len={n}, url ok={len(url)>50}')
    return raw, n, hdr, url

def grab(hdr, n, url, key, outdir='/media/Data-1/npu-direct/hybrid/.st_fix', max_bytes=None):
    v = hdr[key]
    a, b = v['data_offsets']
    if max_bytes: b = min(b, a + max_bytes)
    out = os.path.join(outdir, key + '.bin')
    want = b - a
    if os.path.exists(out) and os.path.getsize(out) == want:
        return out
    base = 8 + n
    subprocess.run(['curl', '-s', '--max-time', '600', '-r', f'{base+a}-{base+b-1}', '-o', out, url], check=True)
    got = os.path.getsize(out)
    print(('OK  ' if got == want else 'BAD '), key, got, '/', want)
    return out

if __name__ == '__main__':
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    keys = sys.argv[2:] if len(sys.argv) > 2 else None
    raw, n, hdr, url = fetch(shard)
    if keys is None:
        keys = [k for k in hdr if k != '__metadata__' and 'layers.3.' in k
                and not ('.experts.' in k and int(k.split('.experts.')[1].split('.')[0]) >= 2)
                and not ('.v_experts.' in k and int(k.split('.v_experts.')[1].split('.')[0]) >= 1)]
    for k in keys:
        grab(hdr, n, url, k)
