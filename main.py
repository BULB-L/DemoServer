import psycopg2 as pg
import os
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np


# 데이터베이스 연결

CONN = pg.connect(
    host="ls-8598f5e3b856b24e455508ee2c5a79fd219ed693.cnqgcgguumqr.ap-northeast-2.rds.amazonaws.com",
    port= 5432,
    database= 'dead_project_snipper',
    user= 'dbmasteruser',
    password= 'BG.7.U>p(6&F3B]*c.*qBWw6Jp`J~~nU',
)


# FastAPI 애플리케이션 생성 port : 
app = FastAPI()

# 루트 엔드포인트 정의
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/projects_list")
def get_projects_list():
    cur = CONN.cursor()
    cur.execute("""
        SELECT project_wallet_address FROM projects
        WHERE project_wallet_address IS NOT NULL and project_wallet_address != '';
        """)
    projects = cur.fetchall()
    cur.close()

    return {'projects_list': [project[0] for project in projects]}

@app.get("/project/github_summary")
def get_project_github_summary(project_wallet_address: str):

    print(project_wallet_address)

    cur = CONN.cursor()
    cur.execute("""
        SELECT * from githubreposmonthlysummary
        WHERE project_wallet_address = %s
        """, (project_wallet_address,))
    
    # key-value 형태로 반환
    
    keys = [desc[0] for desc in cur.description]

    project_github_summary = cur.fetchone()

    if project_github_summary is None:
        return {'project_github_summary': 'the project has no github repos'}

    summary_dict = dict(zip(keys, project_github_summary))

    cur.close()

    return summary_dict

@app.get("/project/all_github_repos_summary")
def get_all_github_repos_summary():

    cur = CONN.cursor()
    cur.execute("""
        SELECT * from githuballreposmonthlysummary order by recorded_at desc limit 1
       """
    )
    
    keys = [desc[0] for desc in cur.description]

    project_github_summary = cur.fetchall()

    summary_dict = [dict(zip(keys, project)) for project in project_github_summary]

    cur.close()

    return summary_dict


@app.get("/project/commits_histo")
def get_project_commits_histo():

    cur = CONN.cursor()
    cur.execute("""
        SELECT monthly_total_commit_lines, monthly_avg_commit_lines from githubreposmonthlysummary
        """
        )
    
    keys = [desc[0] for desc in cur.description]

    project_commit_histo = cur.fetchall()

    # make a df to get hist bins
    df = pd.DataFrame(project_commit_histo, columns=keys)
    total_commit_lines_histbins, total_commit_lines_bin_edges = np.histogram(df['monthly_total_commit_lines'], bins=10)
    avg_total_lines_histbins, avg_total_lines_bin_edges = np.histogram(df['monthly_avg_commit_lines'], bins=10)

    summary_dict = {
        "total_commit_lines" : { 
            "histbins": total_commit_lines_histbins.tolist(),
            'bin_edges': total_commit_lines_bin_edges.tolist(),
        },
        'avg_total_lines' : {
            'histbins': avg_total_lines_histbins.tolist(),
            'bin_edges': avg_total_lines_bin_edges.tolist()
        }   
    }

    cur.close()

    return summary_dict

@app.get("/project/tweets_summary")
def get_project_tweets_summary(project_wallet_address: str):

    cur = CONN.cursor()
    cur.execute("""
        SELECT * from xhandlemonthlysummary
        WHERE project_wallet_address = %s
        """, (project_wallet_address,))
    
    keys = [desc[0] for desc in cur.description]

    project_tweets_summary = cur.fetchall()

    if project_tweets_summary is None:
        return {'project_tweets_summary': 'the project has no tweets'}

    summary_dict = [dict(zip(keys, project)) for project in project_tweets_summary]

    cur.close()

    return summary_dict

@app.get("/project/all_tweets_summary")
def get_all_tweets_summary():

    cur = CONN.cursor()
    cur.execute("""
        SELECT * from xhandleallmonthlysummary order by recorded_at desc limit 1
       """
    )
    
    keys = [desc[0] for desc in cur.description]

    project_tweets_summary = cur.fetchall()

    summary_dict = [dict(zip(keys, project)) for project in project_tweets_summary]

    cur.close()

    return summary_dict

@app.get("/project/tweets_histo")
def get_project_tweets_histo():

    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            monthly_tweet_count, 
            monthly_avg_tweet_likes_count,
            monthly_avg_tweet_retweet_count,
            monthly_avg_tweet_reply_count,
            monthly_avg_tweet_watch_count
        FROM xhandlemonthlysummary
        """
        )
    
    keys = [desc[0] for desc in cur.description]

    project_tweets_histo = cur.fetchall()

    # make a df to get hist bins
    df = pd.DataFrame(project_tweets_histo, columns=keys)
    total_tweets_histbins, total_tweets_bin_edges = np.histogram(df['monthly_tweet_count'], bins=10)
    avg_total_tweets_histbins, avg_total_tweets_bin_edges = np.histogram(df['monthly_avg_tweet_likes_count'], bins=10)
    retweets_histbins, retweets_bin_edges = np.histogram(df['monthly_avg_tweet_retweet_count'], bins=10)
    replies_histbins, replies_bin_edges = np.histogram(df['monthly_avg_tweet_reply_count'], bins=10)
    watch_histbins, watch_bin_edges = np.histogram(df['monthly_avg_tweet_watch_count'], bins=10)

    summary_dict = {
        "monthly_total_tweets" : { 
            "histbins": total_tweets_histbins.tolist(),
            'bin_edges': total_tweets_bin_edges.tolist(),
        },
        'monthly_avg_total_tweets' : {
            'histbins': avg_total_tweets_histbins.tolist(),
            'bin_edges': avg_total_tweets_bin_edges.tolist(),
        },
        'monthly_avg_tweet_retweets_count' : {
            'histbins': retweets_histbins.tolist(),
            'bin_edges': retweets_bin_edges.tolist(),
        },
        'monthly_avg_tweet_replies_count' : {
            'histbins': replies_histbins.tolist(),
            'bin_edges': replies_bin_edges.tolist(),
        },
        'monthly_avg_tweet_watch_count' : {
            'histbins': watch_histbins.tolist(),
            'bin_edges': watch_bin_edges.tolist(),
        }
    }

    cur.close()

    return summary_dict


@app.get("/project/near_txns_summary")
def get_project_near_txns_summary(project_wallet_address: str):

    cur = CONN.cursor()
    cur.execute("""
        SELECT * from wallettxnsmonthlysummary
        WHERE project_wallet_address = %s
        """, (project_wallet_address,))
    
    keys = [desc[0] for desc in cur.description]

    project_near_txns_summary = cur.fetchall()

    if project_near_txns_summary is None:
        return {'project_near_txns_summary': 'the project has no near txns'}

    summary_dict = [dict(zip(keys, project)) for project in project_near_txns_summary]

    cur.close()

    return summary_dict
    

@app.get("/project/all_near_txns_summary")
def get_all_near_txns_summary():

    cur = CONN.cursor()
    cur.execute("""
        SELECT * from wallettxnsallmonthlysummary order by recorded_at desc limit 1
       """
    )
    
    keys = [desc[0] for desc in cur.description]

    project_near_txns_summary = cur.fetchall()

    summary_dict = [dict(zip(keys, project)) for project in project_near_txns_summary]

    cur.close()

    return summary_dict

@app.get("/project/near_txns_histo")
def get_project_near_txns_histo():

    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            total_inbound_transactions,
            total_outbound_transactions,
            avg_inbound_transaction_fee,
            avg_outbound_transaction_fee,
            total_inbound_transaction_fee,
            total_outbound_transaction_fee
        FROM wallettxnsmonthlysummary
        """
        )
    
    keys = [desc[0] for desc in cur.description]

    project_near_txns_histo = cur.fetchall()

    # Make a df to get hist bins
    df = pd.DataFrame(project_near_txns_histo, columns=keys)
    
    # Ensure columns are numeric and handle NaN and inf values explicitly
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, set invalid to NaN
    df = df.replace([np.inf, -np.inf], np.nan)    # Replace inf with NaN
    df = df.fillna(0)                            # Fill NaN with 0

    # Ensure proper data types before passing to np.histogram
    try:
        inbound_total_txns_histbins, inbound_total_txns_bin_edges = np.histogram(df['total_inbound_transactions'], bins=10)
        outbound_total_txns_histbins, outbound_total_txns_bin_edges = np.histogram(df['total_outbound_transactions'], bins=10)    
        inbound_avg_total_txns_histbins, inbound_avg_total_txns_bin_edges = np.histogram(df['avg_inbound_transaction_fee'], bins=10)
        outbound_avg_total_txns_histbins, outbound_avg_total_txns_bin_edges = np.histogram(df['avg_outbound_transaction_fee'], bins=10)
        inbound_total_fee_histbins, inbound_total_fee_bin_edges = np.histogram(df['total_inbound_transaction_fee'], bins=10)
        outbound_total_fee_histbins, outbound_total_fee_bin_edges = np.histogram(df['total_outbound_transaction_fee'], bins=10)
    except Exception as e:
        print("Error during histogram calculation:", e)
        return {"error": str(e)}

    summary_dict = {
        "monthly_total_inbound_txns" : { 
            "histbins": inbound_total_txns_histbins.tolist(),
            'bin_edges': inbound_total_txns_bin_edges.tolist(),
        },
        'monthly_total_outbound_txns' : {
            'histbins': outbound_total_txns_histbins.tolist(),
            'bin_edges': outbound_total_txns_bin_edges.tolist(),
        },
        
        'monthly_avg_inbound_txns' : {
            'histbins': inbound_avg_total_txns_histbins.tolist(),
            'bin_edges': inbound_avg_total_txns_bin_edges.tolist(),
        },
        'monthly_avg_outbound_txns' : {
            'histbins': outbound_avg_total_txns_histbins.tolist(),
            'bin_edges': outbound_avg_total_txns_bin_edges.tolist(),
        },
        'monthly_total_inbound_txns_fee' : {
            'histbins': inbound_total_fee_histbins.tolist(),
            'bin_edges': inbound_total_fee_bin_edges.tolist(),
        },
        'monthly_total_outbound_txns_fee' : {
            'histbins': outbound_total_fee_histbins.tolist(),
            'bin_edges': outbound_total_fee_bin_edges.tolist(),
        }
    }
    cur.close()

    return summary_dict

@app.get("/project/{account}/githubrepos")
def get_project_github_repos(account: str):
    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            githubrepos.github_repo_id,
            githubrepocommitactivitylog.commit_id,
            githubrepocommitactivitylog.commit_total_lines,
            githubrepocommitactivitylog.commit_added_lines,
            githubrepocommitactivitylog.commit_delete_lines,
            githubrepocommitactivitylog.commit_message,
            githubrepocommitactivitylog.commit_date
        from githubrepocommitactivitylog
        right join githubrepos on githubrepocommitactivitylog.github_repo_id = githubrepos.github_repo_id
        WHERE project_wallet_address = %s
        """, (account,))
    
    keys = [desc[0] for desc in cur.description]

    project_github_repos = cur.fetchall()

    if project_github_repos is None:
        return {'project_github_repos': 'the project has no github repos'}

    summary_dict = [dict(zip(keys, project)) for project in project_github_repos]

    cur.close()

    return summary_dict


@app.get("/project/{account}/tweets")
def get_project_tweets(account: str):
    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            xactivitylog.handle,
            xactivitylog.tweet,
            xactivitylog.like_count,
            xactivitylog.retweet_count,
            xactivitylog.reply_count,
            xactivitylog.watch_count,
            xactivitylog.datetime
        from xactivitylog
        right join xhandles on xactivitylog.handle = xhandles.handle
        WHERE xhandles.project_wallet_address = %s
        """, (account,))
    
    keys = [desc[0] for desc in cur.description]

    project_tweets = cur.fetchall()

    if project_tweets is None:
        return {'project_tweets': 'the project has no tweets'}

    summary_dict = [dict(zip(keys, project)) for project in project_tweets]

    cur.close()

    return summary_dict

@app.get("/project/{account}/near_txns")
def get_project_near_txns(account: str):
    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            transaction_hash,
            signer_account_id,
            receiver_account_id,
            block_timestamp,
            actions,
            actions_deposit,
            actions_fee,
            outcomes_status,
            outcomes_transaction_fee,
            burnt_tokens
        from wallettxnslog
        WHERE transaction_id IN (
                select jsonb_array_elements_text(transaction_id_list)
                from projecttxnslist
                where project_wallet_address = %s
        )
        """, (account,))
    
    keys = [desc[0] for desc in cur.description]

    project_near_txns = cur.fetchall()

    if project_near_txns is None:
        return {'project_near_txns': 'the project has no near txns'}

    summary_dict = [dict(zip(keys, project)) for project in project_near_txns]

    cur.close()

    return summary_dict
