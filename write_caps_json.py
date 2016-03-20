import json
import sys

if len(sys.argv) < 4:
    print 'usage: <captions file> <splits file> <output file>'
    exit()
caps_file = sys.argv[1]
splits_file = sys.argv[2]

with open(splits_file,'r') as f:
    image_ids = [int(x[13:-4]) for x in f.read().splitlines()]
with open(caps_file,'r') as f:
    captions = [x for x in f.read().splitlines()]
if len(image_ids) != len(captions):
    print 'numbers of images and captions do not match'
    
    
result = [{'image_id':x, 'caption':y} for (x,y) in zip(image_ids, captions)] 

out_file = sys.argv[3]

with open(out_file,'w') as f:
    json.dump(result, f)
