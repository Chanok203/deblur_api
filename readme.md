# Deblur API

## Runserver

```runserver
pip install -r requirements.txt
unicorn server:app
```

## routes

> /deblur_api/v1/predict

## API Request JSON

```request
{
    "base64_string": "aksdhjnakjsdhasdnalasddsasdlad"
}
```

## API Response JSON

```response
{
    "result": "asdkjnaldjalksdjaksdjlad",
    "time_process": "545.0254",
}
```
