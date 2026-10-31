# WildWatch demo script

About 5 minutes. The dashboard runs from SQLite and saved reports, so only the upload step needs the GPU, and only LLM text for that upload needs internet access.

## Before you start

1. Start the dashboard. It opens http://localhost:8000:

   ```bash
   powershell -ExecutionPolicy Bypass -File src/scripts/run.ps1 dashboard
   ```

2. Check that the home page shows 1,506 images.
3. Keep `docs/reports/screenshots/` open in case the live demo fails.
4. Close other GPU apps, because the pipeline needs the 6 GB card.

Golden and sample images are already in the database, so uploading them only shows "Already analysed". Use **Try a demo photo** on the Upload page, or a file from `data/extra_val/images`. Each demo click uses a different unused real photo, and the results stay in the database.

## 1. The problem (30 seconds)

"Camera traps produce thousands of photos, and most need a person to look at them. Rangers need to know quickly what was seen, how many, and whether a village is at risk."

## 2. Home (60 seconds)

Show the live alert at the top and the 3-tier triage card.

"Every burst gets one of three decisions: auto-discarded, needs review, or kept. We tuned this to avoid losing animals, so only a few bursts are thrown away and doubtful ones go to a person. On the test images we lost 1 of 132 animal bursts."

## 3. Alerts (75 seconds)

Open **Alerts** and show the top alert.

"This is the real photo with the detector's box. The danger score shows all four factors, and the response comes from our config."

Show the label above the message. It says whether the LLM or the template wrote it.

Select **Dispatch team**. "That decision is saved, so it survives a restart." Then select **Reopen** to put it back.

Filter to **Poacher**. Be clear about it: "This also fires on hikers and truck drivers, so it's a prompt for a ranger to check, not an accusation."

Scroll down to **Needs review**: "These are too doubtful to keep or discard, with the most likely animals first."

## 4. Live upload (75 seconds)

Open **Upload** and select **Try a demo photo**, or drop in a new photo plus a broken file.

"The full pipeline runs in a separate process. We don't refresh the page. Each step lights up as the real work reaches it." Results take about a minute.

Show the result: this photo's label, confidence and the model that named it. Select **View alert** if there is one. A broken file shows a clear error. "A bad file never crashes the system."

## 5. Analytics (30 seconds)

Open **Analytics** and show the triage donut, the species split and the cameras table.

"The photos are from 2010 to 2015, so the short time ranges end at the newest photo, and All shows the full history. We sampled day and night evenly, so this isn't natural behaviour."

## 6. Honest accuracy (60 seconds)

Open **Ranger's Brief**. Show that the brief says "Updated just now" after the upload, and ask a question in **Ask the data**. Then scroll down to the test results.

"These results come from test images we opened only once, after freezing everything."

- Species accuracy is 88.3%, with a 95% range of 81.8 to 94.0.
- Precision is 95.4%.
- "We aimed for 90% accuracy and missed it. All 21 missed images are in bursts with no ground-truth box, so they may be label noise, but that isn't proven."
- "Here is a real failure: empty precision on validation falls from 0.94 by day to 0.43 at night."
- Seen and unseen cameras differ by 3.3 points.

## 7. Close (20 seconds)

"WildWatch is a working prototype, trained on US species and with placeholder risk settings. Field use would need local species and real data from rangers."

## Likely questions

| Question | Answer |
|---|---|
| Is this ready for real villages? | No. It covers US species, and the villages and risk factors are placeholders. |
| Does it know Indian animals? | Uploads can name animals from any country. Two small checks of real photos gave 17 of 20 and 13 of 20 hits. Some are marked "possibly": those are low-confidence guesses, and many are wrong. The 88.3% only covers Caltech species. |
| Why not say 96.7%? | That only covers bursts with a confirmed box. The headline number is 88.3%. |
| How accurate is poacher detection? | Not measured, because CCT has no person label. |
| Why are the rare species sightings 0? | The cameras are in the US, so tigers, elephants and leopards aren't expected. |
| How much did the LLM cost? | About $0.008 to generate all messages, against a $15 stop. |
| Why does so much go to review? | We chose not to lose animals. About 15 to 23% of bursts need a person. |

## If something fails

| Problem | What to do |
|---|---|
| Upload takes too long | Show `docs/reports/screenshots/dashboard_upload.png` |
| API is down | Pages show a clear error. Restart with `src/scripts/run.ps1 dashboard` |
| LLM is down | Carry on: alerts use templates and cached text still works |
| Main dashboard won't start | Run `src/scripts/run.ps1 streamlit`, or show the screenshots in `docs/reports/screenshots/` |
