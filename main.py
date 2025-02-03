from functools import wraps
import psycopg2 as pg
import os
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from decimal import Decimal



# 데이터베이스 연결

load_dotenv()

CONN = pg.connect(
    host=os.getenv('DB_HOST'), 
    port=os.getenv('DB_PORT'), 
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
)


def decimal_to_float(value):
    if isinstance(value, Decimal):
        return float(value)
    return value

def ensure_db_connection(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global CONN
        
        def test_connection():
            try:
                # Try a lightweight query to test the connection
                # Use a short timeout to avoid hanging
                CONN.set_session(readonly=True)
                cur = CONN.cursor()
                cur.execute('SELECT 1')
                cur.fetchone()
                CONN.set_session(readonly=False)
                cur.close()
                return True
            except Exception:
                return False
                
        try:
            # First check if connection exists and appears healthy
            if CONN and test_connection():
                return func(*args, **kwargs)
                
            # If we get here, connection is dead or partially alive
            # Close it properly before reconnecting
            if CONN:
                try:
                    CONN.close()
                except Exception:
                    pass  # Ignore errors during close
                    
            # Create new connection
            CONN = pg.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
            )
            return func(*args, **kwargs)
            
        except (pg.OperationalError, pg.InterfaceError):
            # If we still can't connect, try one more time
            if CONN:
                try:
                    CONN.close()
                except Exception:
                    pass
                    
            CONN = pg.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
            )
            return func(*args, **kwargs)
            
    return wrapper


# FastAPI 애플리케이션 생성 port : 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 인증 정보 포함 허용 (예: 쿠키)
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 루트 엔드포인트 정의
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/projects_list")
@ensure_db_connection
def get_projects_list():
    cur = CONN.cursor()
    cur.execute("""
        SELECT project_wallet_address FROM projects
        WHERE project_wallet_address IS NOT NULL and project_wallet_address != '';
        """)
    projects = cur.fetchall()
    cur.close()

    return {'projects_list': [project[0] for project in projects]}

@app.get("/project_info")
@ensure_db_connection
def get_project_info(project_wallet_address: str):
    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            project_potlock_url,
            project_official_website_url,
            created_date
         FROM projects
        WHERE project_wallet_address = %s
        """, (project_wallet_address,))
    project_info = cur.fetchone()

    if project_info is None:
        return JSONResponse({'project_info': 'the project does not exist'}, status_code=404)
    keys = [desc[0] for desc in cur.description]
    project_info = [project for project in project_info]
    project_info_dict = {
        'links' : [project_info[0], project_info[1]],
        'created_date' : project_info[2].strftime('%Y.%m.%d')
    }
    cur.close()

    project_info = dict(zip(keys, project_info))
    
    return JSONResponse(project_info_dict)


@app.get("/project_rank")
@ensure_db_connection
def get_project_rank():
    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            project_wallet_address,
            created_date
        FROM projectreport
        order by score desc
        limit 3
        """
        )
    project_rank = cur.fetchall()



    project_rank = [(project[0], project[1].strftime('%Y.%m.%d')) for project in project_rank]
    
    cur.close()

    return JSONResponse({'project_rank': project_rank})


@app.get("/project_survival_rate")
@ensure_db_connection
def get_project_survival_rate():
    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            active_or_inactive
        FROM projectreport
        where most_recent = true
        """
        )

    project_survival_list = cur.fetchall()
    project_survival_list = [project[0] for project in project_survival_list]

    active_count = project_survival_list.count(True)
    project_survival_rate = active_count / len(project_survival_list)

    return JSONResponse({'project_survival_rate': project_survival_rate})

@app.get("/project/score")
@ensure_db_connection
def get_project_score(project_wallet_address: str):
    cur = CONN.cursor()
    cur.execute("""
        SELECT 
            score
        FROM projectreport
        WHERE project_wallet_address = %s
        """, (project_wallet_address,))
    project_score = cur.fetchone()
    cur.close()

    if project_score is None:
        return JSONResponse({'project_score': 'the project does not exist'}, status_code=404)

    return JSONResponse({'project_score': project_score[0]})

@app.get("/project/github_statistic")
@ensure_db_connection
def get_project_github_statistic(project_wallet_address: str):
    cur = CONN.cursor()
    cur.execute("""
    WITH ranked_data AS (
        SELECT 
            monthly_total_commit_lines,
            monthly_avg_commit_lines,
            PERCENT_RANK() OVER (ORDER BY monthly_total_commit_lines) AS monthly_total_commit_lines_percentile,
            PERCENT_RANK() OVER (ORDER BY monthly_avg_commit_lines) AS monthly_avg_commit_lines_percentile,
            MAX(monthly_total_commit_lines) OVER () AS max_monthly_total_commit_lines,
            MAX(monthly_avg_commit_lines) OVER () AS max_monthly_avg_commit_lines,
            project_wallet_address
        FROM 
            githubreposmonthlysummary
    )
    SELECT 
        monthly_total_commit_lines,
        monthly_avg_commit_lines,
        monthly_total_commit_lines_percentile,
        monthly_avg_commit_lines_percentile,
        max_monthly_total_commit_lines,
        max_monthly_avg_commit_lines
    FROM 
        ranked_data
    WHERE 
        project_wallet_address = %s;

        """, (project_wallet_address,))
    project_github_statistic = cur.fetchone()

        
    if project_github_statistic is None:
        return JSONResponse({'project_github_statistic': 'the project has no github repos'}, status_code=404)

    keys = [desc[0] for desc in cur.description]
    project_github_statistic = dict(zip(keys, project_github_statistic))

    
    # get the all repo statistic
    cur.execute("""
        WITH ranked AS (
            SELECT 
                monthly_total_commit_lines,
                monthly_avg_commit_lines,
                CUME_DIST() OVER (ORDER BY monthly_total_commit_lines) * 100 AS cdf_total_commit_lines,
                CUME_DIST() OVER (ORDER BY monthly_avg_commit_lines) * 100 AS cdf_avg_commit_lines
            FROM GithubReposMonthlySummary
        )
    SELECT 
        (
            SELECT cdf_total_commit_lines FROM ranked 
            WHERE monthly_total_commit_lines >= g.avg_total_commit_lines 
            ORDER BY monthly_total_commit_lines LIMIT 1
        ) AS percentile_avg_total_commit_lines,
        (
            SELECT cdf_avg_commit_lines FROM ranked 
            WHERE monthly_avg_commit_lines >= g.avg_commit_lines 
            ORDER BY monthly_avg_commit_lines LIMIT 1
        ) AS percentile_avg_commit_lines
    FROM GithubAllReposMonthlySummary g;
    """)

    all_repo_statistic = cur.fetchone()

    if all_repo_statistic is None:
        return JSONResponse({'project_github_statistic': 'the project has no github repos'}, status_code=404)

    all_repo_statistic_keys = [desc[0] for desc in cur.description] 
    all_repo_statistic = dict(zip(all_repo_statistic_keys, all_repo_statistic))



    intergrated_statistic = {
        'project': project_github_statistic,
        'all_repo': all_repo_statistic
    }



    cur.close()


    return JSONResponse(intergrated_statistic)


@app.get("/project/github_summary")
@ensure_db_connection
def get_project_github_summary(project_wallet_address: str):



    cur = CONN.cursor()
    cur.execute("""
        SELECT * from githubreposmonthlysummary
        WHERE project_wallet_address = %s
        """, (project_wallet_address,))
    
    # key-value 형태로 반환
    
    keys = [desc[0] for desc in cur.description]

    project_github_summary = cur.fetchone()

    if project_github_summary is None:
        return JSONResponse({'project_github_summary': 'the project has no github repos'}, status_code=404)

    summary_dict = dict(zip(keys, project_github_summary))

    cur.close()

    return summary_dict

@app.get("/project/all_github_repos_summary")
@ensure_db_connection
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
@ensure_db_connection
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


@app.get("/project/tweets_statistic")
@ensure_db_connection
def get_project_tweets_statistic(project_wallet_address: str):
    cur = CONN.cursor()
    cur.execute("""
       WITH ranked_data AS (
    SELECT 
        monthly_tweet_count,
        monthly_avg_tweet_likes_count,
        monthly_avg_tweet_retweet_count,
        monthly_avg_tweet_reply_count,
        monthly_avg_tweet_watch_count,
        PERCENT_RANK() OVER (ORDER BY monthly_tweet_count) AS overall_tweet_count_percentile,
        PERCENT_RANK() OVER (ORDER BY monthly_avg_tweet_likes_count) AS overall_likes_percentile,
        PERCENT_RANK() OVER (ORDER BY monthly_avg_tweet_retweet_count) AS overall_retweet_percentile,
        PERCENT_RANK() OVER (ORDER BY monthly_avg_tweet_reply_count) AS overall_reply_percentile,
        PERCENT_RANK() OVER (ORDER BY monthly_avg_tweet_watch_count) AS overall_watch_percentile,
        MAX(monthly_tweet_count) OVER () AS max_monthly_tweet_count,
        MAX(monthly_avg_tweet_likes_count) OVER () AS max_monthly_avg_likes_count,
        MAX(monthly_avg_tweet_retweet_count) OVER () AS max_monthly_avg_retweet_count,
        MAX(monthly_avg_tweet_reply_count) OVER () AS max_monthly_avg_reply_count,
        MAX(monthly_avg_tweet_watch_count) OVER () AS max_monthly_avg_watch_count,
        project_wallet_address
    FROM 
        xhandlemonthlysummary
)
SELECT 
    monthly_tweet_count,
    monthly_avg_tweet_likes_count,
    monthly_avg_tweet_retweet_count,
    monthly_avg_tweet_reply_count,
    monthly_avg_tweet_watch_count,
    overall_tweet_count_percentile AS monthly_tweet_count_percentile,
    overall_likes_percentile AS monthly_avg_tweet_likes_count_percentile,
    overall_retweet_percentile AS monthly_avg_tweet_retweet_count_percentile,
    overall_reply_percentile AS monthly_avg_tweet_reply_count_percentile,
    overall_watch_percentile AS monthly_avg_tweet_watch_count_percentile,
    max_monthly_tweet_count,
    max_monthly_avg_likes_count,
    max_monthly_avg_retweet_count,
    max_monthly_avg_reply_count,
    max_monthly_avg_watch_count
FROM 
    ranked_data
WHERE 
    project_wallet_address = %s;

        """, (project_wallet_address,))
    project_tweets_statistic = cur.fetchone()

    if project_tweets_statistic is None:
        return JSONResponse({'project_tweets_statistic': 'the project has no tweets'}, status_code=404)

    keys = [desc[0] for desc in cur.description]
    project_tweets_statistic = dict(zip(keys, project_tweets_statistic))

    # get the all repo statistic and percentile

    percentile_sql = """
    WITH ranked AS (
    SELECT 
        monthly_tweet_count,
        monthly_avg_tweet_likes_count,
        monthly_avg_tweet_reply_count,
        monthly_avg_tweet_watch_count,
        monthly_avg_tweet_retweet_count,
        CUME_DIST() OVER (ORDER BY monthly_tweet_count) * 100 AS percentile_tweet_count,
        CUME_DIST() OVER (ORDER BY monthly_avg_tweet_likes_count) * 100 AS percentile_tweet_likes,
        CUME_DIST() OVER (ORDER BY monthly_avg_tweet_reply_count) * 100 AS percentile_tweet_replies,
        CUME_DIST() OVER (ORDER BY monthly_avg_tweet_watch_count) * 100 AS percentile_tweet_watches,
        CUME_DIST() OVER (ORDER BY monthly_avg_tweet_retweet_count) * 100 AS percentile_tweet_retweet
    FROM XHandleMonthlySummary
    )
    SELECT 
        (SELECT percentile_tweet_count FROM ranked 
        WHERE monthly_tweet_count >= a.avg_tweet_count 
        ORDER BY monthly_tweet_count LIMIT 1) AS percentile_avg_tweet_count,
        (SELECT percentile_tweet_likes FROM ranked 
        WHERE monthly_avg_tweet_likes_count >= a.avg_tweet_likes 
        ORDER BY monthly_avg_tweet_likes_count LIMIT 1) AS percentile_avg_tweet_likes,
        (SELECT percentile_tweet_replies FROM ranked 
        WHERE monthly_avg_tweet_reply_count >= a.avg_tweet_replies 
        ORDER BY monthly_avg_tweet_reply_count LIMIT 1) AS percentile_avg_tweet_replies,
        (SELECT percentile_tweet_watches FROM ranked 
        WHERE monthly_avg_tweet_watch_count >= a.avg_tweet_watches 
        ORDER BY monthly_avg_tweet_watch_count LIMIT 1) AS percentile_avg_tweet_watches,
        (SELECT percentile_tweet_retweet FROM ranked 
        WHERE monthly_avg_tweet_retweet_count >= a.avg_tweet_retweet 
        ORDER BY monthly_avg_tweet_retweet_count LIMIT 1) AS percentile_avg_tweet_retweet
    FROM XHandleAllMonthlySummary a;
    
    """

    cur.execute(percentile_sql)

    all_tweet_statistic = cur.fetchone()

    if all_tweet_statistic is None:
        return JSONResponse({'project_tweets_statistic': 'the project has no tweets'}, status_code=404)
    
    all_tweet_statistic_keys = [desc[0] for desc in cur.description] 
    all_tweet_statistic = dict(zip(all_tweet_statistic_keys, all_tweet_statistic))

    intergrated_statistic = {
        'project': project_tweets_statistic,
        'all_handles': all_tweet_statistic
    }

    cur.close()

    return JSONResponse(intergrated_statistic)

@app.get("/project/tweets_summary")
@ensure_db_connection
def get_project_tweets_summary(project_wallet_address: str):

    cur = CONN.cursor()
    cur.execute("""
        SELECT * from xhandlemonthlysummary
        WHERE project_wallet_address = %s
        """, (project_wallet_address,))
    
    keys = [desc[0] for desc in cur.description]

    project_tweets_summary = cur.fetchone()

    if project_tweets_summary is None:
        return JSONResponse({'project_tweets_summary': 'the project has no tweets'}, status_code=404)

    summary_dict = dict(zip(keys, project_tweets_summary))

    cur.close()

    return summary_dict

@app.get("/project/all_tweets_summary")
@ensure_db_connection
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
@ensure_db_connection
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

@app.get("/project/near_txns_statistic")
@ensure_db_connection
def get_project_near_txns_statistic(project_wallet_address: str):
    cur = CONN.cursor()
    cur.execute("""
    WITH ranked_data AS (
        SELECT 
            total_inbound_transactions as monthly_total_inbound_transactions,
            total_outbound_transactions as monthly_total_outbound_transactions,
            avg_inbound_transaction_fee as monthly_avg_inbound_transaction_fee,
            avg_outbound_transaction_fee as monthly_avg_outbound_transaction_fee,
            -- Percentile calculations
            PERCENT_RANK() OVER (ORDER BY total_inbound_transactions) AS monthly_total_inbound_transactions_percentile,
            PERCENT_RANK() OVER (ORDER BY total_outbound_transactions) AS monthly_total_outbound_transactions_percentile,
            PERCENT_RANK() OVER (ORDER BY avg_inbound_transaction_fee) AS monthly_avg_inbound_transaction_fee_percentile,
            PERCENT_RANK() OVER (ORDER BY avg_outbound_transaction_fee) AS monthly_avg_outbound_transaction_fee_percentile,
            -- Maximum value calculations
            MAX(total_inbound_transactions) OVER () AS max_monthly_total_inbound_transactions,
            MAX(total_outbound_transactions) OVER () AS max_monthly_total_outbound_transactions,
            MAX(avg_inbound_transaction_fee) OVER () AS max_monthly_avg_inbound_transaction_fee,
            MAX(avg_outbound_transaction_fee) OVER () AS max_monthly_avg_outbound_transaction_fee,
            project_wallet_address
        FROM 
            wallettxnsmonthlysummary
    )
    SELECT 
        monthly_total_inbound_transactions,
        monthly_total_outbound_transactions,
        monthly_avg_inbound_transaction_fee,
        monthly_avg_outbound_transaction_fee,
        -- Percentiles
        monthly_total_inbound_transactions_percentile,
        monthly_total_outbound_transactions_percentile,
        monthly_avg_inbound_transaction_fee_percentile,
        monthly_avg_outbound_transaction_fee_percentile,
        -- Maximums
        max_monthly_total_inbound_transactions,
        max_monthly_total_outbound_transactions,
        max_monthly_avg_inbound_transaction_fee,
        max_monthly_avg_outbound_transaction_fee
    FROM 
        ranked_data
    WHERE 
        project_wallet_address = %s;

        """, (project_wallet_address,))
    project_near_txns_statistic = cur.fetchone()

    if project_near_txns_statistic is None:
        return JSONResponse({'project_near_txns_statistic': 'the project has no near txns'}, status_code=404)


    keys = [desc[0] for desc in cur.description]
    project_near_txns_statistic = dict(zip(keys, project_near_txns_statistic))

    # get the all repo statistic and percentile

    percentile_sql = """
    WITH ranked AS (
    SELECT 
        total_inbound_transactions,
        total_outbound_transactions,
        avg_inbound_transaction_fee,
        avg_outbound_transaction_fee,
        percent_rank() OVER (ORDER BY total_inbound_transactions) * 100 AS percentile_total_inbound_txns,
        percent_rank() OVER (ORDER BY total_outbound_transactions) * 100 AS percentile_total_outbound_txns,
        percent_rank() OVER (ORDER BY avg_inbound_transaction_fee) * 100 AS percentile_avg_inbound_txn_fee,
        percent_rank() OVER (ORDER BY avg_outbound_transaction_fee) * 100 AS percentile_avg_outbound_txn_fee
    FROM WalletTxnsMonthlySummary
    )
    SELECT 
        (SELECT percentile_total_inbound_txns FROM ranked 
        WHERE total_inbound_transactions >= a.avg_total_inbound_txns 
        ORDER BY total_inbound_transactions LIMIT 1) AS percentile_avg_total_inbound_txns,
        (SELECT percentile_total_outbound_txns FROM ranked 
        WHERE total_outbound_transactions >= a.avg_total_outbound_txns 
        ORDER BY total_outbound_transactions LIMIT 1) AS percentile_avg_total_outbound_txns,
        (SELECT percentile_avg_inbound_txn_fee FROM ranked 
        WHERE avg_inbound_transaction_fee >= a.avg_avg_inbound_txn_fee 
        ORDER BY avg_inbound_transaction_fee LIMIT 1) AS percentile_avg_avg_inbound_txn_fee,
        (SELECT percentile_avg_outbound_txn_fee FROM ranked 
        WHERE avg_outbound_transaction_fee >= a.avg_avg_outbound_txn_fee 
        ORDER BY avg_outbound_transaction_fee LIMIT 1) AS percentile_avg_avg_outbound_txn_fee
    FROM (
        SELECT 
            AVG(total_inbound_transactions) AS avg_total_inbound_txns,
            AVG(total_outbound_transactions) AS avg_total_outbound_txns,
            AVG(avg_inbound_transaction_fee) AS avg_avg_inbound_txn_fee,
            AVG(avg_outbound_transaction_fee) AS avg_avg_outbound_txn_fee
        FROM WalletTxnsMonthlySummary
        GROUP BY month
    ) a;
    """
    cur.execute(percentile_sql)

    all_near_txns_statistic = cur.fetchone()

    if all_near_txns_statistic is None:
        return JSONResponse({'project_near_txns_statistic': 'the project has no near txns'}, status_code=404)

    all_near_txns_statistic_keys = [desc[0] for desc in cur.description] 
    all_near_txns_statistic = dict(zip(all_near_txns_statistic_keys, all_near_txns_statistic))

    cur.close()
    # Convert any Decimal values to float
    all_near_txns_statistic = {k: decimal_to_float(v) for k, v in all_near_txns_statistic.items()}
    project_near_txns_statistic = {k: decimal_to_float(v) for k, v in project_near_txns_statistic.items()}

    intergrated_statistic = {
        'project': project_near_txns_statistic,
        'all_near_txns': all_near_txns_statistic
    }

    return JSONResponse(intergrated_statistic)

@app.get("/project/near_txns_summary")
@ensure_db_connection
def get_project_near_txns_summary(project_wallet_address: str):

    cur = CONN.cursor()
    cur.execute("""
        SELECT * from wallettxnsmonthlysummary
        WHERE project_wallet_address = %s
        """, (project_wallet_address,))
    
    keys = [desc[0] for desc in cur.description]

    project_near_txns_summary = cur.fetchone()

    if project_near_txns_summary is None:
        return JSONResponse({'project_near_txns_summary': 'the project has no near txns'}, status_code=404)

    summary_dict = dict(zip(keys, project_near_txns_summary))

    cur.close()

    return summary_dict
    

@app.get("/project/all_near_txns_summary")
@ensure_db_connection
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
@ensure_db_connection
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
@ensure_db_connection
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
        WHERE githubrepos.project_wallet_address = %s and githubrepocommitactivitylog.commit_date is not null
        """, (account,))
    
    keys = [desc[0] for desc in cur.description]

    project_github_repos = cur.fetchall()
    #make the datetime object to string
    project_github_repos = [list(project) for project in project_github_repos]
    for project in project_github_repos:
        project[6] = project[6].strftime('%Y.%m.%d')

    if not project_github_repos:
        return JSONResponse({'project_github_repos': 'the project has no github repos'}, status_code=404)

    summary_dict = [dict(zip(keys, project)) for project in project_github_repos]

    cur.close()

    return JSONResponse(summary_dict, status_code=200)


@app.get("/project/{account}/tweets")
@ensure_db_connection
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

    # make the datetime object to string
    project_tweets = [list(project) for project in project_tweets]
    for project in project_tweets:
        project[6] = project[6].strftime('%Y.%m.%d')

    if not project_tweets:
        return JSONResponse({'project_tweets': 'the project has no tweets'}, status_code=404)

    summary_dict = [dict(zip(keys, project)) for project in project_tweets]

    cur.close()

    return summary_dict

@app.get("/project/{account}/near_txns")
@ensure_db_connection
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

    if not project_near_txns:
        print('project_near_txns', project_near_txns)
        return JSONResponse({'project_near_txns': 'the project has no near txns'}, status_code=404)

    summary_dict = [dict(zip(keys, project)) for project in project_near_txns]

    cur.close()

    return summary_dict

# get the report of the project
@app.get("/project/github_report")
@ensure_db_connection
def get_github_report(account: str):
    cur = CONN.cursor()
    cur.execute("""
        SELECT report from projectreport
        WHERE project_wallet_address = %s
        """, (account,))
    project_report = cur.fetchone()

    if project_report is None:
        return JSONResponse({'github_report': 'the project has no report'}, status_code=404)
    
    project_report = project_report[0]

    if project_report['has_github'][0] == False:
        return JSONResponse({'github_report': 'the project has no report'}, status_code=404)
    
    if project_report['github_activity_report'] is None:
        return JSONResponse({'github_report': 'the project has no report'}, status_code=404)
    
    if project_report is None:
        return JSONResponse({'github_report': 'the project has no report'}, status_code=404)

    project_report = project_report['github_activity_report'][0]
    # project_report = json.loads(project_report)

    cur.close()


    return JSONResponse(content=project_report)
    
@app.get("/project/tweets_report")
@ensure_db_connection
def get_tweets_report(account: str):
    cur = CONN.cursor()

    cur.execute("""
        SELECT report from projectreport
        WHERE project_wallet_address = %s
        """, (account,))
    project_report = cur.fetchone()

    if project_report is None:
        return JSONResponse({'twitter_report' : 'the project has no report'}, status_code=404)
    
    project_report = project_report[0]

    if project_report['has_twitter'][0] == False:
        return JSONResponse({'twitter_report' : 'the project has no report'}, status_code=404)
    
    if project_report['twitter_activity_report'] is None:
        return JSONResponse({'twitter_report' : 'the project has no report'}, status_code=404)
    
    if project_report is None:
        return JSONResponse({'twitter_report' : 'the project has no report'}, status_code=404)
    
    project_report = project_report['twitter_activity_report'][0]

    cur.close()

    return JSONResponse(content=project_report)
    
@app.get("/project/near_txns_report")
@ensure_db_connection
def get_near_txns_report(account: str):
    cur = CONN.cursor()

    cur.execute("""
        SELECT report from projectreport
        WHERE project_wallet_address = %s
        """, (account,))
    project_report = cur.fetchone()

    if project_report is None:
        return JSONResponse({'near_transaction_report' : 'the project has no report'}, status_code=404)
    
    project_report = project_report[0]

    if project_report['has_near_txns'][0] == False:
        return JSONResponse({'near_transaction_report' : 'the project has no report'}, status_code=404)
    
    if project_report['near_txns_activity_report'] is None:
        return JSONResponse({'near_transaction_report' : 'the project has no report'}, status_code=404)
    
    if project_report is None:
        return JSONResponse({'near_transaction_report' : 'the project has no report'}, status_code=404)
    
    project_report = project_report['near_txns_activity_report'][0]

    cur.close()

    return JSONResponse(content=project_report)

@app.get("/project/overall_report")
@ensure_db_connection
def get_overall_report(account: str):
    cur = CONN.cursor()

    cur.execute("""
        SELECT report from projectreport
        WHERE project_wallet_address = %s
        """, (account,))
    
    project_report = cur.fetchone()

    if project_report is None:
        return JSONResponse({'overall_project_report' : 'the project has no report'}, status_code=404)
    
    project_report = project_report[0]

    project_report = project_report['overall_report']

    cur.close()

    if project_report is None:
        return JSONResponse({'overall_project_report' : 'the project has no report'}, status_code=404)
    
    return JSONResponse(content=project_report)