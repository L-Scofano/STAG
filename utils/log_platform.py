import wandb as wb

class WandbPlatform():
    def __init__(self, name, resume=None, id=None):
        
        self.resume = resume
        
        self.project_name = "contact_for"
        self.group = "luca_exp"
        self.entity = "pinlab-sapienza"
        group = "luca_exp"
        self.group = group
        self.name = name
        self.run = wb.init(
            project=self.project_name,
            entity=self.entity,
            group=self.group,
            name=self.name
            )
        
    def report_scalar(self, name_val, value, iteration = None):

        if iteration is None:
            wb.log({name_val:value})
        else:
            wb.log({name_val:value}, step=iteration)

    def close(self):
        wb.finish()

    def report_args(self, args, name):
        wb.config.update(args)
    
    def get_run_id(self):
        return self.run.id
    
    def get_run_config(self):
        '''
        get_run_config method returns the config attribute of the 
        wandb.run.Run object which is a dictionary of the command
        line arguments and environment variables passed to the run.
        '''
        return self.run.config

    def get_run_summary(self):
        '''
        get_run_summary method returns the summary attribute of 
        the wandb.run.Run object which is a dictionary of the 
        summary metrics that have been logged for the current run.
        '''
        return self.run.summary

    def get_run_step(self):
        '''
        get_run_step method returns the step attribute of the 
        wandb.run.Run object which is the global step of the 
        training process (if the training is using steps).
        '''
        return self.run.step

    def get_run_project(self):
        '''
        get_run_project method returns the project attribute 
        of the wandb.run.Run object which is the name of the 
        project that the run belongs to.
        '''
        return self.run.project