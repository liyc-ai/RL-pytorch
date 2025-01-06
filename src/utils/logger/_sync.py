import getpass
import os
import shutil
from os.path import join
from stat import S_ISDIR as is_remote_dir
from stat import S_ISREG as is_remote_file

import paramiko
from paramiko.sftp_client import SFTPClient

# ========================  Connect  ========================


def connect_remote(
    host: str,
    port: int,
) -> SFTPClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    username = input("Please input your username: ")
    passwd = getpass.getpass("Please input your password: ")

    print(f"Connecting to {username}@{host}:{port}...")
    client.connect(host, port, username, passwd)
    print(f"Successfully connected to {username}@{host}:{port}!")

    return client.open_sftp()


# ========================  Upload  ========================


def _upload_file(sftp: SFTPClient, local_file_path: str, remote_file_path: str):
    sftp.put(local_file_path, remote_file_path)


def _upload_dir(
    sftp: SFTPClient,
    local_log_dir: str,
    local_src_dir: str,
    remote_tgt_dir: str,
    verbose: int = 0,
):
    local_work_dir = join(local_src_dir, local_log_dir)
    remote_work_dir = join(remote_tgt_dir, local_log_dir)

    # If [local_log_dir] exists in remote, we will re-make it
    if local_log_dir in sftp.listdir(remote_tgt_dir):
        sftp.rmdir(remote_work_dir)
    sftp.mkdir(remote_work_dir)

    for item in os.listdir(local_work_dir):
        local_item_path = os.path.join(local_work_dir, item)
        remote_item_path = os.path.join(remote_work_dir, item)

        if os.path.isfile(local_item_path):
            if verbose == 1:  # only report files
                print(f"Uploading {local_item_path} to {remote_item_path}...")
            _upload_file(sftp, local_item_path, remote_item_path)
        elif os.path.isdir(local_item_path):
            if verbose == 2:  # only report dirs
                print(f"Uploading {local_item_path} to {remote_item_path}...")
            _upload_dir(sftp, item, local_work_dir, remote_work_dir, verbose)


def upload_logs(
    host: str,
    port: int,
    local_log_name: str,
    local_src_dir: str,
    remote_tgt_dir: str,
    verbose: int = 0,
):
    """
    Args:
        host: IP address of the remote server
        port: Port of the SSH
        local_log_name: file or directory name
        verbose:
            - 0, not output info during uploading
            - 1, output info of the uploaded files
            - 2, output info of the uploaded directories
    """
    assert verbose in [0, 1, 2], "verbose must only be in [0, 1, 2]"
    local_log_path = os.path.join(local_src_dir, local_log_name)

    sftp = connect_remote(host=host, port=port)

    print(f"Start uploading logs from {local_log_path} to {host}:{port}!")
    if os.path.isfile(local_log_path):
        _upload_file(sftp, local_log_path, os.path.join(remote_tgt_dir, local_log_name))
    else:
        _upload_dir(
            sftp,
            local_log_name,
            local_src_dir,
            remote_tgt_dir,
        )
    print(f"Successfully finish uploading {local_log_path}!")


# ========================  Download  ========================


def _download_file(sftp: SFTPClient, remote_file_path: str, local_file_path: str):
    sftp.get(remote_file_path, local_file_path)


def _download_dir(
    sftp: SFTPClient,
    remote_log_dir: str,
    remote_src_dir: str,
    local_tgt_dir: str,
    verbose: int = 0,
):
    local_work_dir = join(local_tgt_dir, remote_log_dir)
    remote_work_dir = join(remote_src_dir, remote_log_dir)

    # If [local_log_dir] exists in remote, we will re-make it
    if remote_log_dir in os.listdir(local_tgt_dir):
        shutil.rmtree(local_work_dir)
    os.makedirs(local_work_dir)

    for item in sftp.listdir(remote_work_dir):
        local_item_path = os.path.join(local_work_dir, item)
        remote_item_path = os.path.join(remote_work_dir, item)

        item_attr = sftp.lstat(remote_item_path)
        if is_remote_file(item_attr.st_mode):
            if verbose == 1:  # only report files
                print(f"Downloading {remote_item_path} to {local_item_path}...")
            _download_file(sftp, remote_item_path, local_item_path)
        elif is_remote_dir(item_attr.st_mode):
            if verbose == 2:  # only report dirs
                print(f"Downloading {remote_item_path} to {local_item_path}...")
            _download_dir(sftp, item, remote_work_dir, local_work_dir, verbose)


def download_logs(
    host: str,
    port: int,
    remote_log_name: str,
    remote_src_dir: str,
    local_tgt_dir: str,
    verbose: int = 0,
):
    """
    Args:
        host: IP address of the remote server
        port: Port of the SSH
        remote_log_name: file or directory name
        verbose:
            - 0, not output info during downloading
            - 1, output info of the downloaded files
            - 2, output info of the downloaded directories
    """
    assert verbose in [0, 1, 2], "verbose must only be in [0, 1, 2]"
    remote_log_path = os.path.join(remote_src_dir, remote_log_name)

    sftp = connect_remote(host=host, port=port)

    print(f"Start downloading {remote_log_path} from {host}:{port} to {local_tgt_dir}!")
    if os.path.isfile(remote_log_path):
        _download_file(
            sftp, remote_log_path, os.path.join(local_tgt_dir, remote_log_name)
        )
    else:
        _download_dir(
            sftp,
            remote_log_name,
            remote_src_dir,
            local_tgt_dir,
        )
    print(f"Successfully finish downloading {remote_log_path}!")


# # Example: Upload Logs
# load_dotenv("./remote.env")
# """
# Content of remote.env:

# HOSTNAME = "xx.xx.xx.xx"
# PORT = 22
# REMOTE_WORK_DIR = "/path/to/logs"
# """
# upload(
#     hostname=os.environ["HOSTNAME"],
#     port=os.environ["PORT"],
#     local_log_name="logs",
#     local_src_dir="./",
#     remote_tgt_dir=os.environ["REMOTE_WORK_DIR"],
# )
