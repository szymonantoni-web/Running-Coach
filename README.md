# Strava Coach

A Streamlit app that reads your Strava history, takes a race and a target time,
and answers one question: **what should I run next, and why?**

Not a black box. Every recommendation shows the rule that produced it and the
numbers from your own training that triggered it — and if it thinks your goal is
a stretch, it says so rather than telling you what you want to hear.

---

## What it gives you

Several named athletes can each have their own profile — history, goal race and
target time — saved to a Google Sheet you own.

| Tab | What you get |
|---|---|
| **Next session** | The prescribed run — type, distance, structure, target pace — plus the signals that triggered it, this week's volume target, and your acute:chronic load trend |
| **Week ahead** | The next seven days projected out — session, distance and structure for each, rest days placed after the hard ones, plus projected volume, quality count and easy share against target |
| **Add a run** | Drop in a screenshot of a Strava summary; it reads the numbers, shows you what it read, and adds the run only once you have confirmed or corrected it — including what kind of session it was |
| **Fitness & paces** | Your five training paces from a single recent effort, equivalent race times at current fitness, and every run plotted against your pace zones |
| **Goal check** | Current fitness against what the goal requires, whether the timeline covers the gap, and where your volume sits against what that time usually takes |
| **Training history** | Weekly volume, easy share, longest run and load; every run with its classification; CSV export |
| **How it decides** | The full decision tree and, just as importantly, what the tool does *not* know |

---

## The training science behind it

**Fitness → VDOT.** One recent hard effort goes through the Daniels–Gilbert
equations from *Oxygen Power* (1979):

```
%VO2max(T) = 0.8 + 0.1894393·e^(-0.012778·T) + 0.2989558·e^(-0.1932605·T)
VO2(v)     = -4.60 + 0.182258·v + 0.000104·v²
VDOT       = VO2(v) ÷ %VO2max(T)
```

VDOT is an "effective VO2max" inferred from performance rather than measured in a
lab. Every training pace and race prediction follows from that one number. The
test suite checks the implementation against Daniels' published tables — a 19:57
5 km comes back as VDOT 50, which then predicts 41:21 for 10 km and 3:10:49 for
the marathon, matching the book to within a few seconds.

The effort is picked from a **recency window** — 8 weeks by default, adjustable in
the sidebar — and the date it was run is saved with the profile. Both matter:
ranking across a whole imported history surfaces the fastest run of the year, and
a VDOT taken from a months-old effort then propagates into every training pace,
every race prediction and the goal verdict, describing fitness you may no longer
have. On the test data an 8-week window versus a 4-week one is 1.7 VDOT points —
seven minutes of predicted marathon. If a saved effort falls outside the window,
the app says so rather than quietly carrying on.

**What the session was.** Each run carries a session type — Easy, Long run,
Threshold, Intervals, Race and so on — set from a dropdown when you add the run,
and defaulting to a guess from its name, distance and pace. The label overrides
the inference in both directions: an 8 km in 44:00 averages 5:30/km and reads as
easy, but labelled *Intervals* it counts as quality and blocks a second hard
session for three days; a brisk 6 km labelled *Recovery run* stays easy however
fast it was. `Long run` and `Recovery run` likewise override the
distance-based long-run rule.

Leaving the dropdown on *Auto* reproduces the old behaviour exactly, and a
history saved before the column existed loads unchanged, so nothing has to be
relabelled.

**Load → acute:chronic ratio.** Each run scores intensity-weighted kilometres,
`distance × (easy pace ÷ session pace)²`, so an easy kilometre scores 1.0 and
faster running scores more. The last 7 days are compared against the last 28
divided by four. Around 1.0 means this week looks like your recent norm; above
1.5 is the spike pattern worth interrupting.

**Race distance → the shape of the block.** The target distance drives the
coaching layer, not only the VDOT maths. Each distance has a `RaceProfile` in
`plan.py` carrying its phase boundaries, long-run cap, typical volume band and
which quality session carries the block:

| Race | Base / build / peak / taper (weeks out) | Long run cap | Emphasis |
|---|---|---|---|
| Marathon | >18 / 18–8 / 8–3 / <3 | 34 km | Threshold, plus race-pace segments in the long run |
| Half marathon | >14 / 14–6 / 6–2 / <2 | 26 km | Threshold, plus race-pace segments |
| 10 km | >12 / 12–5 / 5–1.5 / <1.5 | 20 km | Threshold and intervals, alternating |
| 5 km | >10 / 10–4 / 4–1 / <1 | 16 km | Intervals, with threshold in support |

This matters more than it looks. A 5 km block sharpens later and tapers for about
a week rather than three, because there is far less accumulated fatigue to shed;
its long runs are capped at 16 km, where a marathon's 34 km would be volume spent
on an adaptation the race does not ask for; and its quality staple is VO2max
interval work, because a 5 km is run close to VO2max while a marathon is run
close to threshold. The volume band is keyed off distance **and** VDOT rather
than finishing time — a 20-minute 5 km and the first 20 minutes of a marathon are
not the same athlete.

**Phase → what kind of session.** Within those boundaries the phase decides
whether quality work means threshold reps, VO2max intervals, or race-pace
segments inside the long run.

**The week ahead.** `week_ahead` projects seven days by running the decision
tree once per day, writing each prescription into a copy of the run history as
though it had been run as given, and deciding the next day from the updated
history. Reusing the real context and load functions rather than approximating
them is the point: a parallel projection drifts out of step with the engine it
is meant to preview.

Building it surfaced two bugs in the engine, which is the usual argument for
simulating a system against itself. A quality session's *average* pace includes
its warm-up and jog recoveries, so a 10 km threshold session read as "easy" and
never reset `days_since_quality` — the projection cheerfully prescribed
threshold four days running. And the long run sits above the quality check in
the tree, so a long run with race-pace segments was being prescribed two days
after a threshold session without anything objecting. Both are fixed, and both
are now tested.

**The decision tree,** in priority order:

| Check | Prescription |
|---|---|
| Quality session within 24 h | Rest or easy shakeout |
| Load ratio above 1.5 | Easy only, until it settles |
| No run for 7+ days | Easy return run |
| No long run for 6+ days | Long run |
| 3+ days since quality, load ≤ 1.3 | Quality session for the phase |
| Otherwise | Easy run, sized to the weekly target |

Every threshold there is a named constant at the top of `plan.py`. They are
conventions, not constants of nature — change them if your coach, your body or
your evidence says otherwise.

---

## Honest limitations

Worth reading before you trust any of it.

- **It does not know how you feel.** Sleep, stress, soreness and illness matter
  more than anything in this repo, and none of them are in a Strava export.
- **It only sees averages.** The bulk export gives one average pace per activity,
  not splits. An interval session shows up at its average pace — which includes
  the jog recoveries — and so looks easier than it was. Two things compensate:
  the classifier reads activity names, and every run you add by hand carries a
  **session type** you pick from a dropdown, which overrides the inference
  outright. Neither is as good as lap data; wire up the API if you want it exact.
- **The ACWR is contested.** It is widely used and widely criticised — the
  original injury-risk findings have not replicated cleanly, and the arithmetic
  is sensitive to how load is defined. It is here as a spike detector, which is
  the part that holds up best.
- **The improvement projection is a planning heuristic**, not a prediction. It
  assumes consistency and no interruptions, and individual response varies far
  more than any single rate captures.
- **Everything here is a population average.** The most useful thing you can do is
  keep your own records and find out where you differ.

- **Two people sharing one sheet see each other's runs within a minute, not
  instantly.** A loaded history is cached for 45 seconds before it is re-read,
  so the Sheets API is not hit on every click. The 🔄 next to the storage line
  in the sidebar forces a re-read immediately. (Before this was added the cache
  was invalidated only by writes from *your own* browser session, so a run added
  anywhere else — the deployed app, a second tab, your laptop — never appeared
  in the others at all.)

Not medical advice. If something hurts in a way that is sharp, one-sided or
getting worse, no algorithm is the right thing to ask.

---

## Run it locally

Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/strava-coach.git
cd strava-coach

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

It opens at `http://localhost:8501`. With nothing saved yet you get a create-a-profile
screen; make one, then import a Strava export into it.

Tests: `pytest`, or `python tests/test_coach.py` and `python tests/test_ocr.py`.

Screenshot reading also needs the Tesseract binary, which is not a Python
package: `brew install tesseract` on macOS, `sudo apt install tesseract-ocr` on
Linux. Without it the app runs fine and everything except the **Add a run** tab
works; the OCR tests skip themselves rather than fail.

---

## Use your own data

**Settings → My Account → Download or Delete Your Account → Request your archive**
on strava.com. The email arrives within a few hours. Upload the `activities.csv`
from inside the zip.

No API keys, no OAuth app registration, no rate limits — which is why this is the
default path. `.gitignore` keeps your own export out of the repo.

The loader is deliberately defensive about that file: Strava writes distance
twice under the same column name (display units and metres), the date format
follows your account locale, and the column set has changed between export
versions. Nothing is read by position — every column is matched by name and units
are sniffed from the data.

**Prefer the API?** `strava.from_api_activities()` takes the JSON list from
`/athlete/activities` and produces the same frame. You will need to register an
app at [strava.com/settings/api](https://www.strava.com/settings/api) and handle
the OAuth exchange; the client secret belongs in `.streamlit/secrets.toml`, which
is git-ignored, never in the code.

---

## The daily loop: adding a run from a screenshot

The CSV export is a one-time job. After that, the **Add a run** tab is the
everyday path: finish a run, screenshot the Strava summary, drop it in, and the
recommendation updates.

**Nothing is saved until you confirm it.** The screenshot is read, the values
appear in an editable form, and you press *Add this run* — so a misread digit
costs you a correction, not a corrupted history.

**How the reading works.** Strava's phone layout is a grid: a big value with a
small label underneath. Read as flat text, neighbouring cells merge — `1:06:52`
and `5:23` come back as `1:06:53:23`, which no regex can untangle. So the reader
finds the label row with Tesseract's word boxes, uses the label positions as
column boundaries, and re-reads each cell cropped on its own. Dark mode is
inverted first, because Tesseract expects dark text on light. Small screenshots
are upscaled, because recognition falls off sharply below about 30 px of text
height.

**The strongest check is arithmetic, not OCR.** Distance, time and pace are
mutually determined, so any two predict the third. If only two are readable the
third is derived; if all three are present and disagree by more than 5%, the app
says so and marks them low-confidence rather than picking a winner. A dropped
decimal point is caught by plausibility bounds — a 124.1 km run or a 1:38/km pace
is flagged, not accepted. That check catches misreads that look perfectly
reasonable on their own, which is exactly the failure mode OCR confidence scores
miss.

**Where it struggles:** heavily cropped screenshots, very small text, and layouts
where values genuinely overflow their columns. In those cases you get a warning
and an editable form rather than a wrong answer — which is the whole design.

**If OCR is unavailable, the tab still works.** No Tesseract binary means no
screenshot reading, but the same form is there to type into — four fields, about
ten seconds. The app tells you why reading is off and how to switch it on rather
than just failing.

### Troubleshooting "Tesseract is not available"

`pytesseract` is only a wrapper; the actual OCR engine is a system binary that
pip cannot install.

- **On Streamlit Community Cloud** — `packages.txt` must be in the **repository
  root**, not a subfolder, containing the single line `tesseract-ocr`. If it is
  already there and the error persists, go to **Manage app → Reboot app**: system
  packages are installed at build time, so an app first built before that file
  existed keeps its old environment until it rebuilds.
- **Locally** — `brew install tesseract` (macOS) or
  `sudo apt install tesseract-ocr` (Linux), then restart Streamlit.
- **Worth knowing:** Community Cloud had a platform-wide outage in early
  September 2026 where `apt-get` failed for *every* app using `packages.txt`
  (expired Debian repository metadata — nothing to do with your file). It was
  fixed on 9 September. If your app was built during that window, a reboot picks
  up the repaired image.

Runs are written to the active profile the moment you confirm them, so there is
nothing to re-upload.

---

## Profiles, and where they are saved

Each profile is one athlete: their run history, goal race, target time, race
date, days per week, and the effort their fitness estimate is based on. Profiles
are **named, not private** — anyone who can open the app can see all of them.
That is deliberate for a couple of people training together; if you need real
separation, restrict the deployed app instead (Streamlit Cloud can limit access
to specific Google accounts).

### Why this needs storage at all

Streamlit Community Cloud's filesystem is **ephemeral**. Anything the app writes
to disk survives until the container restarts — which happens on every redeploy,
every reboot, and whenever the app sleeps and wakes. A local file is a cache, not
storage. So profiles that must survive live in a Google Sheet, outside the app's
lifecycle.

The app detects which it has and says so in the sidebar. With no credentials it
falls back to local files and tells you they are not durable, rather than
quietly losing your data.

### Setting up Google Sheets

About fifteen minutes, once.

1. **Create a spreadsheet** in Google Drive. Call it anything — `strava-coach-data`
   works. Note the key from its URL:
   `docs.google.com/spreadsheets/d/`**`THIS_PART`**`/edit`.
2. At [console.cloud.google.com](https://console.cloud.google.com), create a
   project (or reuse one).
3. **APIs & Services → Library** → enable **Google Sheets API** and **Google
   Drive API**.
4. **APIs & Services → Credentials → Create credentials → Service account.**
   Name it, create it, skip the optional role and user steps.
5. Open the new service account → **Keys → Add key → Create new key → JSON**.
   A file downloads. That file is a password — treat it like one.
6. **Share the spreadsheet with the service account**, as Editor. Its address is
   the `client_email` in the JSON, something like
   `strava-coach@your-project.iam.gserviceaccount.com`. This step is the one
   people miss: a service account is a separate identity and cannot see your
   files just because you own them.
7. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill
   in the values from the JSON plus your spreadsheet key. The `private_key` stays
   on one line with its `\n` escapes exactly as they appear in the JSON.
8. **On Streamlit Cloud** you do not upload that file: paste the same content
   into **Manage app → Settings → Secrets**, then reboot the app.

The two worksheets — `profiles` and `runs` — are created automatically on first
save. They are plain and readable, so you can open the sheet and fix a wrong
number by hand; the app reads columns by header name, so reordering them is
safe. Every save rewrites the sheet rather than patching rows, which at this size
costs nothing and removes a whole category of partial-write bug.

`secrets.toml` is git-ignored. Never commit it.

### If you would rather not use Google

The storage layer is one file with a small interface — `list_profiles`, `load`,
`save`, `delete`. `LocalStore` and `SheetsStore` both implement it and the app
never knows which it has, so adding a third backend (a private Gist, Supabase,
any database) means writing four methods in `storage.py` and nothing else.

---

## Deploy it

[Streamlit Community Cloud](https://share.streamlit.io) hosts it free and
redeploys on every push.

1. Sign in **with GitHub** at share.streamlit.io.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository `<your-username>/strava-coach`, branch `main`, main file `app.py`.
4. **Deploy.**

`packages.txt` must sit in the **repository root** — Community Cloud only looks
there. It installs `tesseract-ocr` at build time, which is what makes the
screenshot tab work in the deployed app. If screenshots come back with a
"Tesseract is not available" error, that file is missing or misplaced.

One thing to think about first: a public app means anyone with the link can
upload a Strava export to it. Nothing is stored server-side — the file lives in
memory for the session and is gone when the tab closes — but if that still feels
wrong, Streamlit Cloud can restrict access to specific Google accounts in the app
settings.

---

## Project layout

```
app.py                       Streamlit interface — layout only, no maths
physiology.py                VDOT, pace zones, race prediction. Pure Python
training_load.py             Weekly volume, intensity split, acute:chronic ratio
plan.py                      Phase logic, the decision tree, goal assessment
strava.py                    Loading and cleaning the export (and the API shape)
ocr.py                       Screenshot reading: Tesseract, then text → structure
storage.py                   Profiles, and the local-file / Google Sheets backends
charts.py                    Altair chart builders
data/sample_activities.csv   Synthetic history the tests run against
tests/test_coach.py          37 tests, including checks against Daniels' tables
tests/test_ocr.py            25 tests, including end-to-end through real Tesseract
tests/test_storage.py        27 tests, both backends through the same assertions
tests/fake_sheets.py         In-memory stand-in for gspread, so storage tests need no network
tests/make_screenshots.py    Renders the Strava-like screenshots the OCR tests use
requirements.txt             Python packages
packages.txt                 System packages — tesseract-ocr, for the screenshot reader
.streamlit/secrets.toml.example   Template for the Google credentials
```

Changing what the coach says means editing `plan.py` and nothing else. Changing
how it looks means `app.py` and nothing else.

## Licence

MIT.
