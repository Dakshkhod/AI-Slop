# TruthLens v4 Deployment Checklist

## Step 1 — On Windows (before Colab)
- [ ] Run organize_data_v4.py — verify zero overlap printed at end
- [ ] Add 20-30 of your own phone photos to D:\data_v4\train\real\
- [ ] Zip D:\data_v4 → right-click → Send to → Compressed folder
- [ ] Locate best_model_v3.pth on your machine

## Step 2 — In Google Colab
- [ ] Runtime → Change runtime type → T4 GPU → Save
- [ ] Paste and run cell1_install.py
- [ ] Paste and run cell2_upload.py — upload data_v4.zip and best_model_v3.pth
- [ ] Paste and run cell3_train.py — ~2.5 hours
- [ ] Paste and run cell4_calibrate.py — ~5 minutes
- [ ] Paste and run cell5_eval.py — write down the numbers
- [ ] Download: best_model_v4.pth, T.json, v4_eval_results.json, training_curves_v4.png

## Step 3 — Deploy
- [ ] Drop best_model_v4.pth and T.json into the backend folder of the repo
- [ ] Verify detector.py has _CKPT_PATH pointing to best_model_v4.pth (already done by Claude Code)
- [ ] git add best_model_v4.pth T.json detector.py
- [ ] git commit -m "v4 model: real photos added, overconfidence fixed, honest eval published"
- [ ] git push — Render auto-redeploys

## Step 4 — Verify after deploy
- [ ] Upload a real phone photo → score should be 65-90, not 99
- [ ] Upload a ChatGPT image → score should be 5-30, not 1
- [ ] Upload a WhatsApp-compressed real photo → should NOT be flagged as AI
- [ ] If any of the above fail, check T.json temperature value — should be > 1.0
