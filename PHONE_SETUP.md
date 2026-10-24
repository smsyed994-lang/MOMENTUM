# Put the dashboard on your phone (about 15 minutes, one time)

**How it works.** GitHub runs your momentum script for free, on a schedule and whenever you tap **Refresh now**.
It saves the results in your repo. Your phone page reads them. No computer needs to stay on.
Your capital, notes and settings stay in your phone's browser.

## One-time setup

**1. Make a GitHub account and a repo**
- Sign up at github.com (free). Tap **New repository**, name it `momentum`, choose **Public**, create it.
  (Public repos get free automation minutes. The repo will show today's stock picks, which come from public
  prices and a public formula. It never contains your capital, notes or token.)

**2. Upload the files**
- Unzip `momentum-phone.zip` on your computer. In the repo, tap **Add file > Upload files**, drag in **everything**
  (including the `.github` folder), then **Commit changes**.
- If the `.github` folder didn't go up: **Add file > Create new file**, type the name
  `.github/workflows/refresh.yml`, paste the contents of that file, and commit.

**3. Run it once**
- Open the **Actions** tab, choose **Refresh momentum data**, tap **Run workflow** (leave the defaults).
- Wait 3 to 5 minutes for a green tick. A `data` folder with `nse.json` / `us.json` appears in the repo.
- If it says permission denied: **Settings > Actions > General > Workflow permissions > Read and write**, then run again.

**4. Turn on the page**
- **Settings > Pages**, under Source choose **Deploy from a branch**, branch `main`, folder `/ (root)`, Save.
- After about a minute your page is at `https://YOURNAME.github.io/momentum/`.

**5. Make a token (lets the Refresh button work)**
- GitHub **Settings > Developer settings > Personal access tokens > Fine-grained tokens > Generate new token**.
- Repository access: **Only select repositories** > `momentum`.
- Permissions: **Contents: Read and write** and **Actions: Read and write**. Generate and copy the token.

**6. Install on your phone**
- Open the page URL in Chrome. Menu (three dots) > **Install app** (or **Add to Home screen**).
- Open the app > **Settings** > enter repo `YOURNAME/momentum` and paste the token > **Save & refresh**.

Done. The market dropdown switches between NSE 500 and S&P 500.

## Everyday use
- The page shows the latest results automatically and rechecks every couple of minutes.
- **Refresh now** asks GitHub to fetch fresh prices; the new numbers arrive in 2 to 5 minutes.
- Scheduled updates run roughly every 30 minutes during market hours (best effort; GitHub can be late).
- **Save official review** records your review (same as the PC button). It is stored in the repo, so your track record is safe.
- Slots, sector cap and weighting are saved to the repo (`config.json`). Capital, currency, FX, alert levels and notes are saved on the phone only.

## Good to know
- **Yahoo can block GitHub's servers.** If a run fails (the page says so), tap Refresh again later. If it keeps failing,
  use the PC route below.
- Notes live in this phone's browser. Clearing site data erases them, and they don't sync to other devices.
- When the token expires the page shows "GitHub said 401". Make a new token and paste it in Settings.
- GitHub can pause scheduled runs in a repo that has been idle for a long time. If updates stop, open Actions and re-enable, or tap Refresh.
- A real `.apk` isn't needed: the installed page opens full screen like an app. If you ever want one, paste your page URL into pwabuilder.com.
- Private repo instead of public? Pages needs a paid plan for private repos, but you can host just `index.html`, the icons, `manifest.webmanifest` and `sw.js` anywhere (for example Netlify or Cloudflare Pages) and keep the data repo private; the token in Settings is then required.

## The PC route (if GitHub can't get Yahoo data)
Run the PC server and open it from your phone on the same Wi-Fi. In Command Prompt, in your project folder:
```
set HOST=0.0.0.0
set APP_PASSWORD=choose-a-password
py app.py
```
On the phone, open `http://YOUR-PC-IP:5000` (find the IP with `ipconfig`) and log in (any user name, your password).
The app refuses to start on the network without a password.
