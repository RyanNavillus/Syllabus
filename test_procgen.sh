export PYTHONWARNINGS="ignore"

echo "Testing cleanrl_procgen with DR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen --curriculum --curriculum-method="dr" --total-timesteps=32768 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen with Async PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen --curriculum --curriculum-method="plr" --total-timesteps=32768 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen with Centralized PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen --curriculum --curriculum-method="centralplr" --total-timesteps=32768 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen with Simple PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen --curriculum --curriculum-method="simpleplr" --total-timesteps=32768 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo -e "Finished testing cleanrl_procgen\n"


echo "Testing cleanrl_procgen_lstm with DR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen_lstm --curriculum --curriculum-method="dr" --total-timesteps=32768 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen_lstm with Async PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen_lstm --curriculum --curriculum-method="plr" --total-timesteps=32768 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen_lstm with Centralized PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen_lstm --curriculum --curriculum-method="centralplr" --total-timesteps=32768 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen_lstm with Simple PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen_lstm --curriculum --curriculum-method="simpleplr" --total-timesteps=32768 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo -e "Finished testing cleanrl_procgen_lstm\n"

echo "Testing cleanrl_procgen_ppg with DR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen_ppg --curriculum --curriculum-method="dr" --total-timesteps=32768 --n-iteration=1 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen_ppg with Async PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen_ppg --curriculum --curriculum-method="plr" --total-timesteps=32768 --n-iteration=1 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen_ppg with Centralized PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen_ppg --curriculum --curriculum-method="centralplr" --total-timesteps=32768 --n-iteration=1 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo "Testing cleanrl_procgen_ppg with Simple PLR"
if ! python -m syllabus.examples.training_scripts.cleanrl_procgen_ppg --curriculum --curriculum-method="simpleplr" --total-timesteps=32768 --n-iteration=1 >/dev/null 2>&1; then
    echo -e "\033[31mFailed\033[0m"
else
    echo -e "\033[32mPassed\033[0m"
fi

echo -e "Finished testing cleanrl_procgen_ppg"