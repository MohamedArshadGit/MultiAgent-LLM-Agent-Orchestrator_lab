# MCP - Multi-Component Python Servers

A set of Python scripts demonstrating basic client-server components.

## Contents

- `client.py` – Connects to servers and sends requests  
- `mathserver.py` – Performs math operations  
- `weatherserver.py` – Provides weather data  

## Usage

Start servers first:

```bash
python mathserver.py
python weatherserver.py
```

Then run the client:

```bash
python client.py
```

*(Update server IP/port in client.py if necessary.)*

## Notes

- Lab/demo setup, not production-ready.  
- Easily extendable with more servers or clients.  

## Author
Mohamed Arshad