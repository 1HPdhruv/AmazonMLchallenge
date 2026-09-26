# EC2 runbook: full-scale runs (r6i.2xlarge, 64 GB)

**Why:** measured locally on 200k real records (reports/phase2_bench.json):
- Representations take about 1.3 KB per record in pandas (about 1.9 KB of process RSS), so 12–20 GB for the
  10.3M-record train pool.
- The sparse TF-IDF matrices add about 2.8 GB.
- The test side needs a comparable amount.

This machine has 15.3 GB in total, so the full run cannot fit locally.

You do steps 1–4 (console sign-in and keys are yours; Claude never handles credentials).

## 1. Launch the instance (AWS console, the region nearest you, e.g. ap-south-1 Mumbai)
- AMI: **Ubuntu Server 24.04 LTS (x86_64)**. Instance type: **r6i.2xlarge** (8 vCPU, 64 GB; about $0.50/h on demand).
  Use r6i.4xlarge (128 GB) if you'd rather have headroom.
- Key pair: create a new one (`amazon-ml.pem`) and keep it out of the repo.
- Network: security group with **SSH (22) from "My IP" only**. Nothing else inbound.
- Storage: **100 GB gp3** root volume (data 2.5 GB + venv + intermediate outputs).
- Advanced → IAM instance profile: only needed for the S3 option in step 4.
- Set a **billing alarm** at, say, $20, and **stop** the instance whenever it is idle (stopped instances don't bill
  compute).

## 2. Connect (Git Bash on Windows)
```bash
chmod 400 ~/Downloads/amazon-ml.pem
ssh -i ~/Downloads/amazon-ml.pem ubuntu@<PUBLIC_DNS>
```

## 3. Bootstrap code + environment + tests (on the instance)
```bash
curl -fsSLO https://raw.githubusercontent.com/1HPdhruv/AmazonMLchallenge/venky/pipeline-v2/tools/ec2_bootstrap.sh \
  || true   # if the repo is private this fails; then scp the script instead (below)
bash ec2_bootstrap.sh
```
If the repo is private:
- Run `scp -i ~/Downloads/amazon-ml.pem tools/ec2_bootstrap.sh ubuntu@<PUBLIC_DNS>:~` from your PC first.
- When `git clone` prompts, use your GitHub username plus a **fine-grained personal access token** (read-only,
  this repo only), not your password.

Expected result: `46 passed`.

## 4. Move the data (never via git; never public)
**Option A: scp (simplest).** From `D:\AmazonMLChallenge2026` in Git Bash:
```bash
tar -czf /d/Dataset.tar.gz Dataset            # ~2.5 GB -> roughly 0.6-0.8 GB
scp -i ~/Downloads/amazon-ml.pem /d/Dataset.tar.gz ubuntu@<PUBLIC_DNS>:~/AmazonMLchallenge/
ssh -i ~/Downloads/amazon-ml.pem ubuntu@<PUBLIC_DNS> 'cd ~/AmazonMLchallenge && tar -xzf Dataset.tar.gz && rm Dataset.tar.gz && ls -la Dataset/*'
```
**Option B: private S3 bucket.**
- Create the bucket with "Block all public access" ON (the default) and no bucket policy.
- Upload from the console. On the instance, attach an IAM role with `s3:GetObject` on that bucket only, then run
  `aws s3 cp s3://<bucket>/Dataset.tar.gz .`
- **Delete the object and the bucket afterwards.**

Then check integrity on both sides:
```bash
sha256sum Dataset/Train/*.tsv Dataset/Test/*.tsv      # instance
```
Compare with the same command in Git Bash locally.

## 5. Runs (Claude drives these over SSH once you share the host, or you run them in tmux)
```bash
cd ~/AmazonMLchallenge && tmux new -s er
unset PYTHONPATH
.venv/bin/python tools/raw_file_audit.py Dataset
.venv/bin/python tools/postings_histogram.py --split train --workers 6
.venv/bin/python tools/phase2_bench.py records --n 200000
.venv/bin/python tools/phase2_bench.py retrieval --s1-frac 0.01 --pools 1000000,2500000,5000000,10320219
```

## 6. Tear-down
- Copy `output/` and `reports/` back with scp. Terminate the instance, delete any S3 object or bucket, and delete
  the key pair if you won't reuse it.
- Nothing from `Dataset/` stays in AWS.
