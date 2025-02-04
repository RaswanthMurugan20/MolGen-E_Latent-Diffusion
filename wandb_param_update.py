import wandb

api = wandb.Api()

# Get the run that you want to change.
run = api.run("raswanth/text_denoising_diffusion/akndljji")

# Change the value of the argument.
run.config[""] = 500

# Update the run.
run.update()
