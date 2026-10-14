# Chef's Hat RL: restricted observations and recurrent memory

**Assigned variant: student ID modulo 7 = 4 (Partial Observability).**

Research question: Is a recurrent actor-critic policy better than a feedforward policy when given limited observations?

This repository uses the [official Chef's Hat Gym environment](https://github.com/pablovin/ChefsHatGYM) and its local four-player room. The [documentation](https://chefshatgym.readthedocs.io/en/latest/) is older API descriptions so the downloaded V3 source was looked at during development. Source is a dependency, not a re-implementation.

## Submission status
- Produce `results/environment_manifest.json` and record exact revision of official source used. The original run was based on a downloaded `main` zip. Its commit hasn't been found yet. Do not claim a precise commit until verified.
- Record and add video viva URL below and confirm assessors have access to repository.
Review and modify the code and the AI statement. Describe each of your submissions in your own words.

**Video viva: pending – add here your accessible 3–5 min recording URL.**

## Design

Chef’s Hat is a turn-based card game. The objective is to be the first to finish. Matches award 3, 2, 1 and 0 points depending upon finishing position; in repeated matches players swap roles and cards.

The policy gets 12 own-hand counts (card ranks 1-12 divided by 17). The legal-action mask. Zero padded hand entries are ignored. The input with the 200-action catalog observed has 212 values. The neural policy does not know the board, public opponent-action events, opponent card counts and role information. The defaults of the random agent are kept by the exchange and special-action decisions.

The legality mask is a network input and an output constraint. Illegal action logits are masked with negative infinity before sampling from a categorical distribution. Actions are using the official ordered catalog and return its integer indices. Both learned agents can choose pass when legal. Valid moves require the mask. Reveal information about the board thereby weakening the constraint.

Both agents use an actor–critic update at the end of a match. The callback gives points so the reward is the difference from the previous total divided by 3. The intermediate rewards are zero. When gamma = 1 every decision in a game gets the terminal return. The actor uses return minus a detached value estimate as its upside. The critic predicts this return.

Loss = mean actor. 0.5*Mean squared value error. 0.01*Mean policy entropy We use Adam with learning rate 0.001 and clip gradients to norm 0.5. Averaging the loss over each match results in matches having equal update weight even when the numbers of decisions differ. This is not PPO or bootstrapped A2C this is monte-carlo actor-critic.

- Feedforward: linear layer to 64 units, tanh then action logits, one value output.

- recurrent: linear layer to 64 units, tanh GRUCell to 64 units, logits and value. The computation graph is preserved throughout the learner’s decisions allowing us to backpropagate through the match. Match. After update memory wipe. The public actions between the turns of the learner do not update the GRU.

The algorithm is suitable for actions and allows the same loss and legality constraints for both architectures. Terminal returns avoid the need for a single-agent Gym step adapter but are noisy in terms of credit assignment and can be inefficient.

## Installation (tested on Windows, python 3.11)

Run commands from the root of this repo in Powershell. Download or clone the exact official source revision used outside of this repo. Leave its source unchanged. 
````powershell py -3.11 -m venv .venv .\.venv\Scripts\python.exe -m install pip --upgrade pip .\.venv\Scripts\python.exe -m pip set up -r requirements.txt ``` `The `requirements.txt` file lists the versions of the dev packages and contains the CPU PyTorch index for `torch==2.14.1+cpu`. These versions are from the student's actual install, it has not been verified if they are available on a fresh machine. The vanilla pip package did not import `core`. You must explicitly specify the official source tree:
````powershell $env:CHEFSHAT_SRC = "C:\path\to\ChefsHatGYM\src" .\.venv\Scripts\python.exe capture_environment.py .\.venv\Scripts\python.exe smoke_test.py ```

Replace the example path with the actual folder where `agents`, `rooms`, and `core` live. `CHEFSHAT_SRC` is for the current terminal; set it again in new terminals. Use the Python 3.11 interpreter from this project in PyCharm, and set CHEFSHAT_SRC in the Run config. A `PYTHONPATH` configuration works for the runners as well but the provenance recorder expects `CHEFSHAT_SRC`.

## Reproduce the experiments

Run 6 new training experiments
``powershell foreach($agentType in ("feedforward","recurrent") ```powershell foreach ($trainingSeed in @(0, 1, 2)) { .\.venv\Scripts\python.exe train.py --agent $agentType --seed $trainingSeed --matches 500 if ($LASTEXITCODE -ne 0) { throw "Training failed, check error." } } ```

Each run saves a checkpoint, history csv and config json, named by architecture, seed and match budget. Training does not overwrite a checkpoint. Save submitted outputs first . Replicate in a new results directory . Filename does not contain learning rate, so do not combine hyperparameter experiments under the same run name.

Assess all 6 checkpoints, then the baselines:
```powershell\.venv\Scripts\python.exe evaluate_all.py .``` shell .venv\Scripts\python.exe evaluate_baselines.py ```

The learned-policy evaluation uses seeds 100, 101, 102, 100 matches per seed, and three random opponents provided. Disables gradients and optimizer updates on loading, checks unchanged weights and action catalog ordering. The evaluation is stochastic, not greedy. Player seats do not rotate and the learner is always given priority. Baselines used the same opponents, match budget per evaluation seed and first-seat position. The given baseline does not skip pass if there is another action, the uniform baseline samples all legal actions.

Evaluation rewrites its CSV output. Back up existing evidence. Re-run. The standard Python, NumPy and PyTorch random generator is seeded. Source inspection shows use of Python `random` for shuffling, starting players, exchanges and random opponents. Equal seeds do not generate identical trajectories over policies, and exact reproducibility across software/platform versions has not been established. The recurrent constructor also consumes random numbers to build an unused feedforward network, so equal seeds do not imply equal initialization.

Regenerate the learning curve from the six training CSVs using the project’s interpreter with `python plot_results.py`. This code checks row counts, score ranges, reward scaling and finite losses before plotting.The last 50 matches with root level pilot checkpoints are `evaluate.py` and `evaluate_feedforward.py`. For the main experiment, use `evaluate_all.py` for 500 matches. `smoke_test.py` just plays a random game, and does not generate learned checkpoints.## Results and interpretation | Policy | Evaluation matches | Average match points (0-3) | Win rate | |---|---:|---:|---:| | Provided random (avoids optional pass) | 300 | 1.540 | 23.3% | Feedforward actor-critic | 900 | 1.304 | 18.1% | | Recurrent actor-critic | 900 | 1.302 | 18.0% | | Uniform legal random | 300 | 0.820 | 8.7% |

The learned-policy totals are from three independently trained checkpoints, each evaluated on 300 matches. Training seeds are not baselines. The evaluation files are `results/evaluation/summary.csv` and `baseline_summary.csv` All rows have the same match budgets. The standard deviations of the mean score at checkpoint level across three training seeds are 0.062 (feedforward) and 0.057 (recurrent). They are descriptive estimates of variation, not confidence intervals or tests of significance. Matches in a repeated game can be correlated by roles and card exchanges.[Individual seeds and their mean 50-match rolling training score](results/learning_curves.png)

The mean training scores for first-100 vs last-100 across training seeds were for feedforward: 0.837 -> 1.203 and for recurrent: 0.730 -> 1.077. Recurrent seed 0, 0.69 to 0.68; its other seeds improved. These changing policy training results show improvement overall, but not a replacement for checkpoint evaluations over training. Pilot runs were setup checks and not used in the main comparison.

Each trained policy tested scored above uniform legal random, below the pass-avoiding random baseline and almost the same as each other. The results show no consistent benefit of recurrent memory under this observation restriction and training budget. No formal statement of importance is made. While optional passing offers a plausible explanation for the baseline gap, pass frequency has not been measured.

## Results and interpretation

| Policy | Evaluation matches | Mean match points (0–3) | Win rate |
|---|---:|---:|---:|
| Provided random (avoids optional pass) | 300 | 1.540 | 23.3% |
| Feedforward actor–critic | 900 | 1.304 | 18.1% |
| Recurrent actor–critic | 900 | 1.302 | 18.0% |
| Uniform legal random | 300 | 0.820 | 8.7% |

The learned-policy totals are from three independently trained checkpoints, each evaluated on 300 matches. Training seeds are not baselines. The evaluation files are `results/evaluation/summary.csv` and `baseline_summary.csv` All rows have the same match budgets. The standard deviations of the mean score at checkpoint level across three training seeds are 0.062 (feedforward) and 0.057 (recurrent). They are descriptive estimates of variation, not confidence intervals or tests of significance. Matches in a repeated game can be correlated by roles and card exchanges.[Individual seeds and their mean 50-match rolling training score](results/learning_curves.png)

The mean training scores for first-100 vs last-100 across training seeds were for feedforward: 0.837 -> 1.203 and for recurrent: 0.730 -> 1.077. Recurrent seed 0, 0.69 to 0.68; its other seeds improved. These changing policy training results show improvement overall, but not a replacement for checkpoint evaluations over training. Pilot runs were setup checks and not used in the main comparison.

Each trained policy tested scored above uniform legal random, below the pass-avoiding random baseline and almost the same as each other. The results show no consistent benefit of recurrent memory under this observation restriction and training budget. No formal statement of importance is made. While optional passing offers a plausible explanation for the baseline gap, pass frequency has not been measured.

### Environment performance-score discrepancy

Older docs describe a round-adjusted performance score . In the downloaded V3 source, the room's game-over payload contains `final_scores`, and `Game.step` invokes `dataset.end_experiment(..., self.scores.copy(), self.scores.copy())`. So the dataset's performance field receives the same sum of points. This project explicitly reports the mean match points and win rate and does not re-label it as the older round-adjusted metric. Source provenance and an assessor’s interpretation of the required metric still matter.

## Limitations and improvements

- The legal action mask shows the restrictions from the board and removing the board might not remove much useful information.
- The GRU has access only to the learner’s own decision inputs, not all public events. Memory might be more useful if events were better coded.
- Network sizes 200 actions feedforward 26697 trainable parameters recurrent 51657 trainable parameters . The experiment does not distinguish memory from capacity.
- Only three training seeds, 500 games each, one learning rate and one opponent strategy was tested. More seeds/budgets, different opponents, hyperparameter sweeps, and seat rotation would help strengthen conclusions.
- No advantage normalization, terminal-only returns and full-match recurrent graphs can lead to noisy updates and poor credit assignment. Another optimization option is match-length averaging.
- Rewards are for each match even if roles impact subsequent matches. The policies do not optimize a formally modeled long horizon objective over the entire repeated game.
- We did not ablate unrestricted observations, learned-opponent training, pass-frequency analysis or untrained-network checkpoint baseline Uniform random is an action baseline, not an initial network baseline.
- Source revision not pinned yet. The runtime fingerprint logs bytes but doesn't restore a missing upstream revision.

## Video viva outline

Use the 3-5 minutes to explain the environment and variant, show restricted inputs and legal masking, explain reward differences and the actor-critic update, demonstrate a frozen trained agent, show the learning curve and evaluation table, and discuss the lack of memory advantage and baseline gap. Do not show random play as an example of a trained agent. Evaluate with the 500-match checkpoints using `evaluate_all.py` or a separately configured frozen single-checkpoint evaluator.
## AI use declaration

CChatGPT was beneficial in Windows/PyCharm setup, agent and experiment-runner code creating hint, debugging of the cumulative-score reward errors, recurrent memory and frozen evaluation code creating, CSV analysis, visualization, my written code review. Some areas has highlighted AI-assisted code. The student is responsible for reviewing, adapting, checking and understanding all submitted content. This structure does not pretend to be independently developed or completely understood. ( These numbers weren't created , they have been calculated from files generated by regular execution on the student 's PC ) . The experimental CSVs . I update this according to the task that has been done.

## References

- Official source https://github.com/pablovin/ChefsHatGYM
- Documentation: https://chefshatgym.readthedocs.io/en/latest/
- PyTorch installation guide: https://pytorch.org/get-started/locally/
