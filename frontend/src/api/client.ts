const configuredBaseUrl =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export const API_BASE_URL = configuredBaseUrl.replace(/\/$/, '')

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

interface ApiErrorBody {
  detail?: string
}

export async function apiRequest<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...init?.headers,
      },
    })
  } catch {
    throw new ApiError(
      'Could not reach the SignalBridge API. Confirm the backend is running.',
      0,
    )
  }

  if (!response.ok) {
    let message = `The API returned status ${response.status}.`
    try {
      const body = (await response.json()) as ApiErrorBody
      if (body.detail) message = body.detail
    } catch {
      // Keep the safe status-based message when the body is not JSON.
    }
    throw new ApiError(message, response.status)
  }

  return (await response.json()) as T
}
