git remote add -f -t master --no-tags kilt-wikipedia https://github.com/huggingface/datasets.git

git rm -rf --ignore-unmatch kilt_wikipedia/
git read-tree --prefix=kilt_wikipedia/ -u kilt-wikipedia/master:datasets/kilt_wikipedia
echo "overwrote kilt_wikipedia with latest version, you might want to 'git commit'"
