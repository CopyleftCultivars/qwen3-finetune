import json
from cactus.bindings import cactus as c
m=c.cactus_init('/work/bundle')
texts=["I am in Washington State and grow heirloom corn","One fact about soil:","How do I increase potassium in my garden?"]
out={t:c.cactus_embed(m,t,normalize=True) for t in texts}
json.dump(out, open('/work/out/engine_embed_6L.json','w'))
print("dim", len(out[texts[0]]))
c.cactus_destroy(m)
