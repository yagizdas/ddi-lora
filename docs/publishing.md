# Publish on GitHub

Suggested repository name: **patient-aware-ddi**.

Suggested description: **DDInter-grounded synthetic data generation and Qwen3-14B LoRA fine-tuning for patient-aware drug interaction research.**

Suggested topics: `nlp`, `lora`, `qwen3`, `synthetic-data`, `drug-drug-interactions`, `machine-learning`, `research`.

The local repository is initialized on `main`. No commit, GitHub remote, or upload is created by the cleanup. Create an empty repository on GitHub, then run from this folder:

```bash
python scripts/check_publication.py
git add .
git diff --cached --stat
git status --short
git commit -m "Prepare patient-aware DDI research repository"
git remote add origin https://github.com/YOUR_USERNAME/patient-aware-ddi.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If you initialize the GitHub repository with a README, license, or other commit, reconcile that history before pushing instead of overwriting it.

The publication check scans files Git considers publishable, checks common token patterns, confirms ignored local folders and credentials, and verifies clean notebooks. It is a useful check, not a guarantee that every possible private value is recognized. Review the staged file list before pushing.

The public report includes the author's academic contact information as supplied in the PDF. The complete datasets, duplicate submission folders, original notebooks with outputs, archives, trained weights, and local `.env` are excluded by `.gitignore`.

Choose an open-source license if you want to grant reuse rights, after checking the attribution notes. No license was selected on the author's behalf.

For your resume, a defensible description is:

> Built a DDInter-grounded synthetic data pipeline and LoRA-adapted Qwen3-14B for patient-aware interaction assessment; reported 13/15 passing evaluation scenarios versus 9/15 for the baseline while updating approximately 0.86% of model parameters.

Link the repository and keep the small evaluation scope visible. The result does not establish clinical validity or production deployment.
