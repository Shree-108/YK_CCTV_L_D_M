"""CPU cost of one YOLO11s training step (forward+backward+SGD step) on RANDOM tensors.
No dataset, no pretrained weights, nothing saved: compute/memory sizing only."""
import time, json, os, gc
import psutil, torch
from ultralytics.nn.tasks import DetectionModel

torch.manual_seed(0)
proc = psutil.Process(os.getpid())
print(json.dumps(dict(cpu_threads=torch.get_num_threads(), logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(False),
                      ram_total_gb=round(psutil.virtual_memory().total / 2**30, 1), ram_free_gb=round(psutil.virtual_memory().available / 2**30, 1))))
res = []
for imgsz, batch, freeze in ((640, 8, 10), (640, 8, 0), (960, 4, 10), (1280, 4, 10), (1280, 2, 10), (1280, 4, 0)):
    m = DetectionModel("yolo11s.yaml", nc=1, verbose=False); m.train()
    for i, layer in enumerate(m.model):
        if i < freeze:
            for p in layer.parameters(): p.requires_grad_(False)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-3, momentum=0.9)
    x = torch.randn(batch, 3, imgsz, imgsz)
    gc.collect(); base = proc.memory_info().rss; peak = base; ts = []
    for it in range(4):
        t0 = time.perf_counter()
        out = m(x)
        def leaves(o):
            if torch.is_tensor(o): yield o
            elif isinstance(o, dict):
                for v in o.values(): yield from leaves(v)
            elif isinstance(o, (list, tuple)):
                for v in o: yield from leaves(v)
        loss = sum(t.float().pow(2).mean() for t in leaves(out) if t.requires_grad)
        opt.zero_grad(); loss.backward(); opt.step()
        ts.append(time.perf_counter() - t0); peak = max(peak, proc.memory_info().rss)
    s_img = sorted(ts[1:])[1] / batch
    m.eval()
    with torch.no_grad():
        xi = torch.randn(1, 3, imgsz, imgsz); m(xi); t0 = time.perf_counter(); m(xi); inf = time.perf_counter() - t0
    r = dict(imgsz=imgsz, batch=batch, freeze=freeze, train_s_per_img=round(s_img, 2), infer_s_per_img=round(inf, 2),
             peak_rss_gb=round(peak / 2**30, 2),
             est_epoch_min_50img=round((50 * s_img + 50 * inf) / 60, 1))
    res.append(r); print(json.dumps(r), flush=True)
    del m, opt, x; gc.collect()
