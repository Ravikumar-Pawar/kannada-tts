# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (2 min)
python src/data_prep.py

# 3. TEST INFERENCE NOW ✅ (30 sec)
python src/inference.py

# 4. Train later
# python src/train_tacotron.py



# 1. Fix Git (5 sec)
git config core.autocrlf false
git rm --cached -r .
git reset --hard

# 2. Create .gitattributes (permanent fix)
echo "* text=auto eol=lf`n*.bat text eol=crlf" > .gitattributes
git add .gitattributes

# 3. Add all files
git add .
git commit -m "Complete project setup + fix line endings"

# 4. CONTINUE TTS WORK ✅
pip install -r requirements.txt
python src/data_prep.py
python src/inference.py

