"""
Move all Drive .tif files matching a name pattern into a folder, then download them.

Usage:
    python gdrive_move_and_download.py
"""

import json
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Config ────────────────────────────────────────────────────────────────────
NAME_CONTAINS   = "N03E033_ERA5_TEMP"
DRIVE_FOLDER    = "ERA5_LAND_TEMP"
LOCAL_DIR       = Path(r"D:\flood_data\N03\N03E033\ERA5_LAND_TEMP")


# Reuse the Earth Engine OAuth credentials (they already carry the Drive scope).
EE_CRED_PATHS = [
    Path.home() / ".config" / "earthengine" / "credentials",
    Path.home() / "AppData" / "Roaming" / "earthengine" / "credentials",
]
TOKEN_URI = "https://oauth2.googleapis.com/token"
# ─────────────────────────────────────────────────────────────────────────────


def get_service():
    cred_path = next((p for p in EE_CRED_PATHS if p.exists()), None)
    if cred_path is None:
        raise FileNotFoundError(
            "Earth Engine credentials not found. Run `earthengine authenticate` first."
        )

    data = json.loads(cred_path.read_text())
    if "https://www.googleapis.com/auth/drive" not in data.get("scopes", []):
        raise PermissionError(
            "EE credentials lack the Drive scope. Re-run `earthengine authenticate` "
            "with Drive access enabled."
        )

    creds = Credentials(
        token=None,
        refresh_token=data["refresh_token"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        token_uri=TOKEN_URI,
        scopes=data["scopes"],
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, folder_name: str) -> str:
    """Return the Drive folder ID, creating it if it doesn't exist."""
    query = (
        f"name = '{folder_name}' "
        "and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])
    if folders:
        folder_id = folders[0]["id"]
        print(f"Found existing Drive folder '{folder_name}' (id={folder_id})")
        return folder_id

    meta = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    folder = service.files().create(body=meta, fields="id").execute()
    folder_id = folder["id"]
    print(f"Created Drive folder '{folder_name}' (id={folder_id})")
    return folder_id


def search_tif_files(service, name_contains: str) -> list[dict]:
    """Return all .tif files whose name contains name_contains."""
    query = (
        f"name contains '{name_contains}' "
        "and mimeType != 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    files, page_token = [], None
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, parents)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def move_file(service, file_id: str, target_folder_id: str, current_parents: list[str]):
    service.files().update(
        fileId=file_id,
        addParents=target_folder_id,
        removeParents=",".join(current_parents),
        fields="id, parents",
    ).execute()


def download_file(service, file_id: str, dest_path: Path):
    request = service.files().get_media(fileId=file_id)
    with dest_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    service = get_service()

    print(f"Searching Drive for files containing '{NAME_CONTAINS}'...")
    files = search_tif_files(service, NAME_CONTAINS)
    print(f"Found {len(files)} files.")

    if not files:
        print("Nothing to do.")
        return

    folder_id = get_or_create_folder(service, DRIVE_FOLDER)

    already_downloaded = {p.name for p in LOCAL_DIR.glob("*.tif")}
    moved = skipped = downloaded = 0

    for i, f in enumerate(files, 1):
        name      = f["name"]
        file_id   = f["id"]
        parents   = f.get("parents", [])
        dest_path = LOCAL_DIR / name

        # Move to target folder if not already there
        if folder_id not in parents:
            move_file(service, file_id, folder_id, parents)
            moved += 1
        else:
            skipped += 1

        # Download if not already present
        if name in already_downloaded:
            print(f"  [{i}/{len(files)}] Skip (exists): {name}")
        else:
            print(f"  [{i}/{len(files)}] Downloading: {name}")
            download_file(service, file_id, dest_path)
            downloaded += 1

    print(f"\nDone. Moved: {moved}  Already in folder: {skipped}  Downloaded: {downloaded}")


if __name__ == "__main__":
    main()
