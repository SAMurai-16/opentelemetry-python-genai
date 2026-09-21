# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Amazon Bedrock Agent Runtime invoke_agent API instrumentation."""

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from opentelemetry.instrumentation.genai.bedrock.extractors import (
    extract_invoke_agent_request,
)
from opentelemetry.instrumentation.genai.bedrock.patch import (
    _handle_invoke_agent,
)
from opentelemetry.instrumentation.genai.bedrock.stream import (
    BedrockInvokeAgentStreamWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler


def _mock_make_request(
    client, events, status_code=200, session_id="session-123"
):
    """Mock the underlying endpoint.make_request for bedrock-agent-runtime client."""
    http_response = MagicMock(status_code=status_code, headers={})
    parsed_response = {
        "completion": events,
        "contentType": "application/json",
        "sessionId": session_id,
        "ResponseMetadata": {"HTTPStatusCode": status_code, "HTTPHeaders": {}},
    }
    client._endpoint.make_request = MagicMock(
        return_value=(http_response, parsed_response)
    )


def test_invoke_agent_with_content(
    bedrock_agent_runtime_client,
    instrument_with_content,
    span_exporter,
) -> None:
    events = [
        {"chunk": {"bytes": b"Hello from "}},
        {"chunk": {"bytes": b"Bedrock Agent!"}},
    ]
    _mock_make_request(
        bedrock_agent_runtime_client, events, session_id="session-123"
    )

    response = bedrock_agent_runtime_client.invoke_agent(
        agentId="AGENT001",
        agentAliasId="ALIAS001",
        sessionId="session-123",
        inputText="Hello agent",
    )

    assert isinstance(response["completion"], BedrockInvokeAgentStreamWrapper)
    stream_events = list(response["completion"])
    assert len(stream_events) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "invoke_agent"
    assert span.kind == SpanKind.CLIENT
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_AGENT_ID] == "AGENT001"
    assert span.attributes[GenAIAttributes.GEN_AI_AGENT_VERSION] == "ALIAS001"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID]
        == "session-123"
    )
    assert (
        span.attributes[ServerAttributes.SERVER_ADDRESS]
        == "bedrock-agent-runtime.us-east-1.amazonaws.com"
    )
    assert span.attributes[ServerAttributes.SERVER_PORT] == 443
    assert span.status.status_code == StatusCode.UNSET

    input_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["content"] == "Hello agent"

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["parts"][0]["content"] == "Hello from Bedrock Agent!"


def test_invoke_agent_no_content(
    bedrock_agent_runtime_client,
    instrument_no_content,
    span_exporter,
) -> None:
    events = [
        {"chunk": {"bytes": b"Secret agent response"}},
    ]
    _mock_make_request(
        bedrock_agent_runtime_client, events, session_id="session-456"
    )

    response = bedrock_agent_runtime_client.invoke_agent(
        agentId="AGENT002",
        agentAliasId="ALIAS002",
        sessionId="session-456",
        inputText="Secret input",
    )

    assert isinstance(response["completion"], BedrockInvokeAgentStreamWrapper)
    stream_events = list(response["completion"])
    assert len(stream_events) == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "invoke_agent"
    assert span.kind == SpanKind.CLIENT
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_AGENT_ID] == "AGENT002"
    assert span.attributes[GenAIAttributes.GEN_AI_AGENT_VERSION] == "ALIAS002"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID]
        == "session-456"
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert span.status.status_code == StatusCode.UNSET


def test_invoke_agent_non_chunk_events(
    bedrock_agent_runtime_client,
    instrument_with_content,
    span_exporter,
) -> None:
    events = [
        {"trace": {"trace": {}}},
        {"chunk": {"bytes": b"Actual answer"}},
        {"files": {"files": []}},
    ]
    _mock_make_request(
        bedrock_agent_runtime_client, events, session_id="session-789"
    )

    response = bedrock_agent_runtime_client.invoke_agent(
        agentId="AGENT003",
        agentAliasId="ALIAS003",
        sessionId="session-789",
        inputText="Test non chunk events",
    )

    stream_events = list(response["completion"])
    assert len(stream_events) == 3

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["parts"][0]["content"] == "Actual answer"


def test_invoke_agent_stream_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.invoke_remote_agent("aws.bedrock")

    def failing_generator():
        yield {"chunk": {"bytes": b"part 1"}}
        raise ConnectionError("Stream disconnected")

    wrapper = BedrockInvokeAgentStreamWrapper(
        failing_generator(), invocation=invocation
    )

    with pytest.raises(ConnectionError, match="Stream disconnected"):
        list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


def test_invoke_agent_direct_call_error(
    bedrock_agent_runtime_client,
    instrument_with_content,
    span_exporter,
) -> None:
    error_response = {
        "Error": {
            "Code": "ResourceNotFoundException",
            "Message": "Agent not found",
        }
    }
    bedrock_agent_runtime_client._endpoint.make_request = MagicMock(
        side_effect=ClientError(error_response, "InvokeAgent")
    )

    with pytest.raises(ClientError):
        bedrock_agent_runtime_client.invoke_agent(
            agentId="INVALID_AGENT",
            agentAliasId="ALIAS",
            sessionId="session-err",
            inputText="hello",
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] in (
        "ResourceNotFoundException",
        "botocore.errorfactory.ResourceNotFoundException",
        "botocore.exceptions.ClientError",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_AGENT_ID] == "INVALID_AGENT"


def test_handle_invoke_agent_without_completion(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    mock_instance = MagicMock()
    mock_instance.meta.endpoint_url = (
        "https://bedrock-agent-runtime.us-east-1.amazonaws.com"
    )
    mock_wrapped = MagicMock(return_value={"sessionId": "session-no-stream"})
    api_params = {
        "agentId": "AGENT_SYNC",
        "agentAliasId": "ALIAS_SYNC",
        "sessionId": "session-no-stream",
        "inputText": "Sync test",
    }

    response = _handle_invoke_agent(
        mock_wrapped,
        mock_instance,
        ("InvokeAgent", api_params),
        {},
        api_params,
        handler,
    )

    assert response == {"sessionId": "session-no-stream"}
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "invoke_agent"
    assert spans[0].attributes[GenAIAttributes.GEN_AI_AGENT_ID] == "AGENT_SYNC"


def test_extract_invoke_agent_request_without_inputText(
    tracer_provider,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.invoke_remote_agent("aws.bedrock")

    api_params = {
        "agentId": "AGENT_NO_INPUT",
        "agentAliasId": "ALIAS_NO_INPUT",
        "sessionId": "session-no-input",
    }
    extract_invoke_agent_request(api_params, invocation, capture_content=True)

    assert invocation.agent_id == "AGENT_NO_INPUT"
    assert invocation.agent_version == "ALIAS_NO_INPUT"
    assert invocation.conversation_id == "session-no-input"
    assert invocation.input_messages == []
    invocation.stop()
