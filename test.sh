python3 convert.py -s datasets/isaac_south/
python3 export_image_embeddings.py --checkpoint ckpts/sam_vit_h_4b8939.pth --model-type vit_h --input ../../datasets/isaac_south/images/ --output ../../datasets/isaac_south/sam_embeddings
python3 train.py -s data/isaac_south/ -m output/isaac_south -f sam -r 0 --speedup --iterations 7000

