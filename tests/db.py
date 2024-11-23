import pyhadro
import numpy as np

pyhadro.message("Starting test")

db = pyhadro.Database('test')

rng = np.random.default_rng(5891)

N = 1000
for i in range(N):
    y = rng.normal(1, 0.1*np.sqrt(N), size=(100,))
    db.add_data(y, f"test", "config", f"c-{i}")

db.jackknife(('test', 'config'), ('test', 'jk_mean'), lambda x: x)

x, y, ys = db.curve('test', 'jk_mean')

redchi2 = sum((y - 1)**2/ys**2)/100
assert redchi2 < 2

db.save()
