from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "diffusion_policy"

    args.abs_actions = True

    generator = get_generator(
        algo_name="diffusion_policy",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/diffusion_policy.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[8],
    )

    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[1000],
    )

    # use ddim by default
    generator.add_param(
        key="algo.ddim.enabled",
        name="ddim",
        group=1001,
        values=[
            True,
            # False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="algo.ddpm.enabled",
        name="ddpm",
        group=1001,
        values=[
            False,
            # True,
        ],
        hidename=True,
    )

    ### Multi-task training on atomic tasks ###
    EVAL_TASKS = ["CloseDrawer", "CloseSingleDoor"] # or evaluate all tasks by setting EVAL_TASKS = None
    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=[
            (get_ds_cfg("single_stage", src="human", eval=EVAL_TASKS, filter_key="50_demos"), "human-50"), # training on human datasets
            (get_ds_cfg("single_stage", src="mg", eval=EVAL_TASKS, filter_key="3000_demos"), "mg-3000"), # training on MimicGen datasets
        ]
    )

    generator.add_param(
        key="train.action_keys",
        name="ac_keys",
        group=-1,
        values=[
            [
                "action_dict/abs_pos",
                "action_dict/abs_rot_6d",
                "action_dict/gripper",
                "action_dict/base_mode",
                # "actions",
            ],
        ],
        value_names=[
            "abs",
        ],
        hidename=True,
    )
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)