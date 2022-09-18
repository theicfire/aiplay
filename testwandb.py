import wandb
import time

import bpdb; bpdb.set_trace()
wandb.init(project="my-test-project")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}


for i in range(100):
    loss = 100 - i
    print('fake loss', loss)
    wandb.log({"loss": loss})
    time.sleep(1)

# Optional
# wandb.watch(model)
