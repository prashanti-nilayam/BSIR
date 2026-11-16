"""Direction definitions: the four panels that make up each contrast.

A direction d = (defender D, peer P, task). Each has four matched panels:
  d_homo  -- [D,D,D]        defender's own homogeneous panel
  p_homo  -- [P,P,P]        peer's own homogeneous panel
  mixed   -- [D,D,P]        honest heterogeneous
  adv     -- [D,D,P-adv]    peer assigned the adversarial role

Matched support is the set of (replicate, item) pairs present with all agents
in R0 and R1 of all four panels.
"""

def _r(stem, reps=(1, 2, 3)):
    return [stem.format(rep=r) for r in reps]


DIRECTIONS = [
    dict(key="gemma12b_qwen7b_math", label="Gemma-3-12B <- Qwen-7B",
         task="MATH-hard", benchmark="math",
         d_homo=_r("honest_gemma3_homo_math_5round_rep{rep}"),
         p_homo=_r("honest_qw7b_homo_math_5round_rep{rep}"),
         mixed=_r("het_2g1qw_honest_math_5round_rep{rep}"),
         adv=_r("adv_2g1qwadv_math_5round_rep{rep}")),

    dict(key="qwen7b_llama8b_gsm8k", label="Qwen-7B <- Llama-8B",
         task="GSM8K", benchmark="gsm8k",
         d_homo=_r("qwen7b_homo_gsm8k_5round_rep{rep}"),
         p_homo=_r("llama8b_homo_gsm8k_5round_rep{rep}"),
         mixed=_r("het_2qq1ll_honest_gsm8k_5round_rep{rep}"),
         adv=_r("adv_2qq1lladv_gsm8k_5round_rep{rep}")),

    dict(key="gemini_gpt41_math", label="Gemini-Flash <- GPT-4.1",
         task="MATH-hard", benchmark="math",
         d_homo=_r("scaled_e1_gemini_math_hard_5round_4k_rep{rep}"),
         p_homo=_r("scaled_e1_math_hard_5round_n500_4k_rep{rep}"),
         mixed=_r("mixed_2gem1gpt_e1_math_hard_5round_n500_4k_rep{rep}"),
         adv=_r("adv_2gem1gpt_e1_math_hard_5round_n500_4k_rep{rep}")),

    dict(key="gpt41_gemini_math", label="GPT-4.1 <- Gemini-Flash",
         task="MATH-hard", benchmark="math",
         d_homo=_r("scaled_e1_math_hard_5round_n500_4k_rep{rep}"),
         p_homo=_r("scaled_e1_gemini_math_hard_5round_4k_rep{rep}"),
         mixed=_r("mixed_2gpt1gem_e1_math_hard_5round_n500_4k_rep{rep}"),
         adv=_r("adv_2gpt1gem_e1_math_hard_5round_n500_4k_rep{rep}")),

    dict(key="gpt41_gptoss_math", label="GPT-4.1 <- OSS-120B",
         task="MATH-hard", benchmark="math",
         d_homo=_r("scaled_e1_math_hard_5round_n500_4k_rep{rep}"),
         p_homo=_r("scaled_e1_gptoss120b_math_hard_5round_4k_rep{rep}"),
         mixed=_r("honest_2gpt1oss_e1_math_hard_5round_n500_4k_rep{rep}"),
         adv=_r("adv_2gpt1oss_e1_math_hard_5round_n500_4k_rep{rep}")),

    dict(key="llama8b_qwen7b_gsm8k", label="Llama-8B <- Qwen-7B",
         task="GSM8K", benchmark="gsm8k",
         d_homo=_r("llama8b_homo_gsm8k_5round_rep{rep}"),
         p_homo=_r("qwen7b_homo_gsm8k_5round_rep{rep}"),
         mixed=_r("het_2ll1qw7b_gsm8k_5round_rep{rep}"),
         adv=_r("adv_2ll1qw7badv_gsm8k_5round_rep{rep}")),

    dict(key="qwen72b_gptoss_math", label="Qwen-72B <- OSS-120B",
         task="MATH-hard", benchmark="math",
         d_homo=_r("qwen72b_homo_e1_math_hard_5round_rep{rep}"),
         p_homo=_r("scaled_e1_gptoss120b_math_hard_5round_4k_rep{rep}"),
         mixed=_r("honest_2qw1oss_e1_math_hard_5round_rep{rep}"),
         adv=_r("adv_2qw1ossadv_e1_math_hard_5round_rep{rep}")),

    dict(key="llama1b_gemma2b_gsm8k", label="Llama-1B <- Gemma-2B",
         task="GSM8K-diag", benchmark="gsm8k",
         d_homo=_r("homo_3llama_e1_gsm8k_full_rep{rep}", reps=(1,)),
         p_homo=_r("gemma2b_gsm8k_cap", reps=(1,)),
         mixed=_r("honest_2llama1gemma_e1_gsm8k_5round_rep{rep}", reps=(1,)),
         adv=_r("adv_2llama1gemma_e1_gsm8k_5round_rep{rep}", reps=(1,))),

    dict(key="llama70b_gptoss_math", label="Llama-70B <- OSS-120B",
         task="MATH-hard", benchmark="math",
         d_homo=_r("scaled_e1_llama70b_math_hard_5round_n500_4k_rep{rep}"),
         p_homo=_r("scaled_e1_gptoss120b_math_hard_5round_4k_rep{rep}"),
         mixed=_r("honest_2llama1oss_e1_math_hard_5round_n500_4k_rep{rep}"),
         adv=_r("adv_2llama1oss_e1_math_hard_5round_n500_4k_rep{rep}")),

    # Fixed-receiver directions: Qwen-7B held as defender, peer varied.
    dict(key="qwen7b_gemma12b_math", label="Qwen-7B <- Gemma-3-12B",
         task="MATH-hard-fx", benchmark="math", fixed_receiver=True,
         d_homo=_r("honest_qw7b_homo_hard_5round_rep{rep}"),
         p_homo=_r("honest_gemma3_homo_math_5round_rep{rep}"),
         mixed=_r("het_2qw1g_honest_hard_5round_rep{rep}"),
         adv=_r("adv_2qw1gadv_hard_5round_rep{rep}")),

    dict(key="qwen7b_gptoss_math", label="Qwen-7B <- OSS-120B",
         task="MATH-hard-fx", benchmark="math", fixed_receiver=True,
         d_homo=_r("honest_qw7b_homo_hard_5round_rep{rep}"),
         p_homo=_r("scaled_e1_gptoss120b_math_hard_5round_4k_rep{rep}"),
         mixed=_r("het_2qw7b1oss_honest_hard_5round_rep{rep}"),
         adv=_r("adv_2qw7b1ossadv_hard_5round_rep{rep}")),
]

DEFENDER_SLOTS = (0, 1)   # slots holding D in every condition
PEER_SLOT = 2             # slot holding P in the mixed and adversarial panels
