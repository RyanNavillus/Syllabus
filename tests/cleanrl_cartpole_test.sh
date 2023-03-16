# python3 -m tests.cleanrl_utils.benchmark \
#     --env-ids CartPole-v1 \
#     --command "python3 -m syllabus.examples.cleanrl_cartpole --exp-name cartpole_curriculum --track" \
#     --num-seeds 5 \
#     --workers 5 \
#     --auto-tag False

# python3 -m tests.cleanrl_utils.benchmark \
#     --env-ids CartPole-v1 \
#     --command "python3 -m syllabus.examples.cleanrl_cartpole_rs5 --exp-name cartpole_curriculum_rs5 --track" \
#     --num-seeds 5 \
#     --workers 5 \
#     --auto-tag False

# python3 -m tests.cleanrl_utils.benchmark \
#     --env-ids CartPole-v1 \
#     --command "python3 -m syllabus.examples.cleanrl_cartpole_rs5_sp01_steps10 --exp-name cartpole_curriculum_rs5_sp01_steps10 --track" \
#     --num-seeds 5 \
#     --workers 5 \
#     --auto-tag False

# python3 -m tests.cleanrl_utils.benchmark \
# --env-ids CartPole-v1 \
# --command "python3 -m syllabus.examples.cleanrl_cartpole_rs5_sp75_steps10 --exp-name cartpole_curriculum_rs5_sp75_steps10 --track" \
# --num-seeds 5 \
# --workers 5 \
# --auto-tag False

# python3 -m tests.cleanrl_utils.benchmark \
# --env-ids CartPole-v1 \
# --command "python3 -m syllabus.examples.cleanrl_cartpole_dumb --exp-name cartpole_curriculum_dumb --track" \
# --num-seeds 5 \
# --workers 5 \
# --auto-tag False

python3 -m tests.cleanrl_utils.benchmark \
--env-ids CartPole-v1 \
--command "python3 -m syllabus.examples.cleanrl_cartpole_rs5_sp75_steps10_fast --exp-name cartpole_curriculum_rs5_sp75_steps10_fast --track" \
--num-seeds 5 \
--workers 5 \
--auto-tag False

# python3 -m tests.cleanrl_utils.benchmark \
#     --env-ids CartPole-v1 \
#     --command "python3 -m syllabus.examples.cleanrl_cartpole_nocurr --exp-name cartpole --track" \
#     --num-seeds 5 \
#     --workers 5 \
#     --auto-tag False

