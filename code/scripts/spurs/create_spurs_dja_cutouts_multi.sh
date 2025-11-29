for f in f070w-clear \
         f090w-clear \
         f115w-clear \
         f140m-clear \
         f150w-clear \
         f182m-clear \
         f200w-clear \
         f210m-clear \
         f250m-clear \
         f277w-clear \
         f300m-clear \
         f335m-clear \
         f356w-clear \
         f360m-clear \
         f410m-clear \
         f430m-clear \
         f444w-clear \
         f460m-clear \
         f480m-clear
do
    python create_spurs_dja_cutouts_single.py --filter "$f"
done