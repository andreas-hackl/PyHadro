import pyhadro

db = pyhadro.Database('test')
db.print()

d = pyhadro.gpt_io.get_data("test.dat")
