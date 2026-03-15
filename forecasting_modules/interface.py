import time, copy, gc, re, pandas as pd, numpy as np, matplotlib.pyplot as plt
import os.path; from datetime import datetime, timedelta

from data_collection_modules import OpenMeteo
from forecasting_modules.tasks import ForecastingTaskSingleTarget
from data_collection_modules.eu_locations import (
    countries_metadata
)
from data_modules.data_loaders import (
    extract_from_database,
    clean_and_impute
)

from logger import get_logger
logger = get_logger(__name__)

def main_forecasting_pipeline(c_dict:dict,task_list:list, outdir:str, database:str, freq:str, verbose:bool):

    if not os.path.isdir(outdir):
        if verbose: logger.info("Creating output directory {}".format(outdir))
        os.makedirs(outdir)

    if not freq in ['hourly', 'minutely_15']:
        raise ValueError(f"freq must be one of 'hourly', 'minutely_15' Given {freq}")

    for task in task_list:
        run_label = task['label']

        logger.info(f" <<<<<<<< Running {run_label}  >>>>>>>>>>>>>>")

        # get features + target (historic) and features (forecast) from database
        df_hist, df_forecast = extract_from_database(
            main_pars=task, c_dict=c_dict, db_path=database, outdir=outdir, verbose=verbose, freq=freq,
        )
        task['targets'] = [str(col) for col in task['targets'] if col in df_hist.columns.tolist()]


        # clean data from nans and outliers
        df_hist, df_forecast = clean_and_impute(df_hist=df_hist,df_forecast=df_forecast,freq=freq,verbose=verbose)

        # initialize the processor for tasks
        processor = ForecastingTaskSingleTarget(
            df_history=df_hist,df_forecast=df_forecast,task=task,outdir=outdir,verbose=verbose
        )

        # process task to fine-tune the forecasting model. Note: ensemble tasks require base models to be processed first
        if task['task_fine_tuning']:
            for ft_task in task['task_fine_tuning']:
                logger.info(f" <<<<<<<< Running Finetuning Task for {ft_task['model']}  >>>>>>>>>>>>>>")
                if ft_task['model'].__contains__('ensemble'):
                    processor.process_finetuning_task_ensemble(ft_task)
                else:
                    processor.process_finetuning_task_base(ft_task)

        # train forecasting model on full dataset assuming hyperparameters are in finetuning dir
        if task['task_training']:
            for t_task in task['task_training']:
                logger.info(f" <<<<<<<< Running Training Task for {t_task['model']}  >>>>>>>>>>>>>>")
                if t_task['model'].__contains__('ensemble'):
                    processor.process_training_task_ensemble(t_task)
                else:
                    processor.process_training_task_base(t_task)

        # forecast with trained model
        if task['task_forecasting']:
            for f_task in task['task_forecasting']:
                logger.info(f" <<<<<<<< Running Forecasting Task for {f_task['model']}  >>>>>>>>>>>>>>")
                if f_task['model'].__contains__('ensemble'):
                    processor.process_forecasting_task_ensemble(f_task)
                else:
                    processor.process_forecasting_task_base(f_task)

        if task['task_plot']:
            logger.info(f" <<<<<<<< Running Plotting >>>>>>>>>>>>>>")
            processor.process_task_plot_predict_forecast(task)

        if task['task_summarize']:
            logger.info(f" <<<<<<<< Running Summarization Task >>>>>>>>>>>>>>")
            processor.process_task_determine_the_best_model(task, outdir=outdir+run_label+'/')

        if task.get('task_evaluation'):
            for e_task in task['task_evaluation']:
                logger.info(f" <<<<<<<< Running Evaluation Task for {e_task['model']}  >>>>>>>>>>>>>>")
                if 'ensemble' in e_task['model']:
                    processor.process_evaluation_task_ensemble(e_task)
                else:
                    processor.process_evaluation_task_base(e_task)

        processor.clean()

        gc.collect()

