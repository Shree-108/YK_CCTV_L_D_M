# CCTV integration status and vendor dependencies

The full details, test results and the questions for CP PLUS are in **`docs/CCTV_RECORDING_RETRIEVAL.md`**.

| Capability | Status |
|---|---|
| Live RTSP view / scheduled screenshots (code exists) | **Never tested on the real NVR.** No NVR address, credentials or confirmed RTSP path |
| Search recordings by camera and time on the NVR | **Unverified, not implemented.** Only retailer claims of ONVIF Profile G; no official CP PLUS documentation |
| Automatic download of recordings with Python | **Unverified, not implemented** |
| Manual export in KVMS Pro Lite, then **automatic processing** of the export folder | **Implemented and tested** with real store footage in clips with synthetic names (`ingest/export_folder.py`): de-duplication, camera from the folder, start time from the file name, timestamps, outputs, network-failure recovery. **Not yet tested with a real KVMS export file** |

Vendor and installer dependencies:

1. **CP PLUS:** written confirmation of ONVIF Profile G, SDK or playback-RTSP support for CP-UNR-104F1 on our
   firmware, and the permitted use.
2. **Installer / IT:** NVR LAN/VPN reachability from the AI PC (no port forwarding) and a view-only/playback account.
3. **Site:** one sample KVMS Pro Lite export (MP4/AVI) with its original file name, so the name pattern and time zone
   can be configured.

No NVR settings were changed, and the NVR was never contacted.
