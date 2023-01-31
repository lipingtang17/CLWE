# Downloading all dictionaries tested in the paper
cd data/dictionaries/
for src_lg in en es de it ms he ar ja zh
do
  for tgt_lg in en es de it ms he ar ja zh
  do
    if [ $src_lg != $tgt_lg ]
    then
      for suffix in .5000-6500.txt
      do
        fname=$src_lg-$tgt_lg$suffix
        wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$fname
      done
    fi
  done
done
