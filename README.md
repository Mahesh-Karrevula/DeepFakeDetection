"""Run the trained detector on one video:  python infer.py path/to/video.mp4
Pipeline: 16 frames -> JPEG re-encode -> MTCNN face crop -> CLIP ViT-B/16 (layer 8 + last CLS, mean over frames)
          + AST audio embedding -> two MLP heads -> fake if either fires.
NOTE: written to match the training pipeline but not verified end to end. Test on a few FakeAVCeleb videos first."""
import sys, os, json, numpy as np, cv2, torch, torch.nn as nn, av
from facenet_pytorch import MTCNN
from transformers import CLIPVisionModel, ASTFeatureExtractor, ASTModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
N, SIZE, SR, MAXS = 16, 224, 16000, 163840
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

def mlp(d):
    return nn.Sequential(nn.Linear(d, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
                         nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

heads = torch.load(os.path.join(D, "heads.pt")); nz = np.load(os.path.join(D, "norm.npz"))
cfg = json.load(open(os.path.join(D, "config.json")))
hv, ha = mlp(1536), mlp(768)
hv.load_state_dict(heads["video"]); ha.load_state_dict(heads["audio"]); hv.eval(); ha.eval()

mtcnn = MTCNN(keep_all=False, select_largest=True, device=dev, post_process=False)
clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).to(dt).eval()
fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).to(dt).eval()
cm = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
cs = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)

def frames(path, q=60):
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(np.linspace(0, total - 1, N).astype(int).tolist()); fr, i = [], 0
    while len(fr) < N:
        if not cap.grab(): break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                f = cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
                fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release(); assert fr, "could not read video"
    while len(fr) < N: fr.append(fr[-1])
    return fr[:N]

def crops(fr, margin=1.3):
    H, W = fr[0].shape[:2]; sel = [fr[0], fr[N // 2], fr[-1]]; sc = min(1.0, 640 / W)
    small = np.stack([cv2.resize(f, (int(W * sc), int(H * sc))) for f in sel])
    boxes, _ = mtcnn.detect(small); found = [b[0] / sc for b in boxes if b is not None]
    if found:
        b = np.stack(found); x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; side = min(margin * max(x1 - x0, y1 - y0), min(H, W))
    else:
        print("warning: no face found, using center crop"); cx, cy, side = W / 2, H / 2, min(H, W)
    xs = int(np.clip(cx - side / 2, 0, W - side)); ys = int(np.clip(cy - side / 2, 0, H - side)); s = int(side)
    return np.stack([cv2.resize(f[ys:ys + s, xs:xs + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in fr])

def audio(path):
    out, n = [], 0
    with av.open(path) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for frame in c.decode(audio=0):
            res = rs.resample(frame)
            for f in (res if isinstance(res, list) else [res]):
                a = f.to_ndarray().reshape(-1); out.append(a); n += len(a)
            if n >= MAXS: break
    return np.concatenate(out)[:MAXS].astype(np.float32)

@torch.no_grad()
def predict(path):
    x = torch.from_numpy(crops(frames(path))).to(dev).permute(0, 3, 1, 2).float() / 255.0
    hs = clip(pixel_values=((x - cm) / cs).to(dt), output_hidden_states=True).hidden_states
    v = torch.cat([hs[8][:, 0], hs[-1][:, 0]], 1).float().cpu().numpy().mean(0)
    a = ast(input_values=fe([audio(path)], sampling_rate=SR, return_tensors="pt")["input_values"].to(dev).to(dt)
            ).pooler_output.float().cpu().numpy()[0]
    pv = torch.sigmoid(hv(torch.from_numpy(((v - nz["mu_v"]) / nz["sd_v"]).astype(np.float32))[None])).item()
    pa = torch.sigmoid(ha(torch.from_numpy(((a - nz["mu_a"]) / nz["sd_a"]).astype(np.float32))[None])).item()
    return pv, pa, 1 - (1 - pv) * (1 - pa)

if __name__ == "__main__":
    pv, pa, p = predict(sys.argv[1])
    print(f"video-fake: {pv:.3f} | audio-fake: {pa:.3f} | overall fake score: {p:.3f}")
    print("VERDICT:", "FAKE" if p >= cfg["threshold"] else "REAL", f"(threshold {cfg['threshold']})")