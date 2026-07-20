import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'

import { ApiError } from '../api/client'
import { uploadTranscript } from '../api/transcripts'
import { Breadcrumbs } from '../components/Breadcrumbs'

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function validateFile(file: File | null): string | null {
  if (!file) return 'Select a transcript file before uploading.'
  if (!file.name.toLowerCase().endsWith('.txt')) {
    return 'Only .txt transcript files are supported. Choose a plain-text transcript and try again.'
  }
  if (file.size === 0) {
    return 'The selected transcript file is empty. Choose a non-empty .txt file.'
  }
  return null
}

export function UploadTranscript() {
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropZoneRef = useRef<HTMLDivElement>(null)
  const successRef = useRef<HTMLDivElement>(null)
  const dragDepthRef = useRef(0)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [isDragActive, setIsDragActive] = useState(false)

  const uploadMutation = useMutation({
    mutationFn: uploadTranscript,
    onSuccess: () => {
      setValidationError(null)
      queryClient.invalidateQueries({ queryKey: ['transcripts'] })
    },
  })

  useEffect(() => {
    if (uploadMutation.isSuccess) successRef.current?.focus()
  }, [uploadMutation.isSuccess])

  function selectFile(file: File | null) {
    const error = validateFile(file)
    setValidationError(error)
    setSelectedFile(error ? null : file)
    if (error && fileInputRef.current) fileInputRef.current.value = ''
    uploadMutation.reset()
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0] ?? null)
  }

  function openFilePicker() {
    if (!uploadMutation.isLoading) fileInputRef.current?.click()
  }

  function removeFile() {
    if (uploadMutation.isLoading) return
    setSelectedFile(null)
    setValidationError(null)
    uploadMutation.reset()
    if (fileInputRef.current) fileInputRef.current.value = ''
    dropZoneRef.current?.focus()
  }

  function handleDropZoneKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openFilePicker()
    }
  }

  function handleDragEnter(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    if (uploadMutation.isLoading) return
    dragDepthRef.current += 1
    setIsDragActive(true)
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    event.dataTransfer.dropEffect = uploadMutation.isLoading ? 'none' : 'copy'
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) setIsDragActive(false)
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    dragDepthRef.current = 0
    setIsDragActive(false)
    if (uploadMutation.isLoading) return

    if (event.dataTransfer.files.length !== 1) {
      setSelectedFile(null)
      setValidationError('Choose exactly one .txt transcript per upload.')
      uploadMutation.reset()
      return
    }

    selectFile(event.dataTransfer.files[0])
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const error = validateFile(selectedFile)
    if (error) {
      setValidationError(error)
      return
    }
    uploadMutation.mutate(selectedFile as File)
  }

  const apiError =
    uploadMutation.error instanceof ApiError
      ? uploadMutation.error.message
      : uploadMutation.isError
        ? 'An unexpected error occurred while uploading the transcript.'
        : null

  return (
    <section className="upload-page" aria-labelledby="upload-title">
      <Breadcrumbs
        items={[
          { label: 'Transcripts', to: '/' },
          { label: 'Upload transcript' },
        ]}
      />

      <div className="page-header upload-page-header">
        <div className="page-header__content">
          <p className="eyebrow">New transcript</p>
          <h2 className="page-title" id="upload-title">Upload transcript</h2>
          <p className="page-description">
            Add one meeting transcript to the workspace. Analysis starts only
            when you initiate it from the transcript record.
          </p>
        </div>
      </div>

      {uploadMutation.isSuccess ? (
        <div
          className="upload-complete surface"
          ref={successRef}
          role="status"
          tabIndex={-1}
        >
          <div className="upload-complete__mark" aria-hidden="true">✓</div>
          <p className="eyebrow">Upload complete</p>
          <h3>{uploadMutation.data.file_name}</h3>
          <p>
            The transcript is stored and ready for speaker-turn review and
            processing. Analysis has not started.
          </p>
          <dl>
            <div>
              <dt>Status</dt>
              <dd className="status-badge status-badge--ready">
                {uploadMutation.data.status.replace(/_/g, ' ')}
              </dd>
            </div>
            <div>
              <dt>Token count</dt>
              <dd>{uploadMutation.data.token_count.toLocaleString()}</dd>
            </div>
          </dl>
          <div className="upload-complete__actions">
            <Link
              className="button button--primary"
              to={`/transcripts/${uploadMutation.data.id}`}
            >
              View transcript
            </Link>
            <Link className="button button--secondary" to="/">
              Back to transcripts
            </Link>
          </div>
        </div>
      ) : (
        <div className="upload-workspace">
          <form className="upload-form surface" onSubmit={handleSubmit} noValidate>
            <input
              className="visually-hidden"
              id="transcript-file"
              ref={fileInputRef}
              name="file"
              type="file"
              accept=".txt,text/plain"
              aria-label="Choose a transcript file"
              onChange={handleFileChange}
              disabled={uploadMutation.isLoading}
            />

            <div
              ref={dropZoneRef}
              className={`upload-drop-zone${isDragActive ? ' is-drag-active' : ''}${uploadMutation.isLoading ? ' is-disabled' : ''}`}
              role="button"
              tabIndex={uploadMutation.isLoading ? -1 : 0}
              aria-controls="transcript-file"
              aria-disabled={uploadMutation.isLoading}
              onClick={openFilePicker}
              onKeyDown={handleDropZoneKeyDown}
              onDragEnter={handleDragEnter}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <span className="upload-drop-zone__icon" aria-hidden="true">↑</span>
              <strong>
                {isDragActive ? 'Release to select this transcript' : 'Drop a .txt transcript here'}
              </strong>
              <span>or choose a file</span>
              <small>One non-empty UTF-8 .txt file</small>
            </div>

            {selectedFile ? (
              <div className="selected-file-card" aria-live="polite">
                <div className="selected-file-card__details">
                  <span className="selected-file-card__status">Valid .txt file</span>
                  <strong>{selectedFile.name}</strong>
                  <span>{formatFileSize(selectedFile.size)}</span>
                </div>
                <div className="selected-file-card__actions">
                  <button
                    className="button button--ghost"
                    type="button"
                    disabled={uploadMutation.isLoading}
                    onClick={openFilePicker}
                  >
                    Choose another
                  </button>
                  <button
                    className="button button--ghost selected-file-card__remove"
                    type="button"
                    disabled={uploadMutation.isLoading}
                    onClick={removeFile}
                  >
                    Remove
                  </button>
                </div>
              </div>
            ) : null}

            {validationError ? (
              <div className="alert alert--error upload-alert" role="alert">
                <strong>File not accepted</strong>
                <p>{validationError}</p>
              </div>
            ) : null}

            {apiError ? (
              <div className="alert alert--error upload-alert" role="alert">
                <strong>Upload could not be completed</strong>
                <p>{apiError}</p>
              </div>
            ) : null}

            {uploadMutation.isLoading ? (
              <div className="upload-progress" role="status" aria-live="polite">
                <span className="spinner" aria-hidden="true" />
                <span>Uploading transcript. Analysis has not started.</span>
              </div>
            ) : null}

            <button
              className="button button--primary upload-submit"
              type="submit"
              disabled={!selectedFile || uploadMutation.isLoading}
            >
              {uploadMutation.isLoading ? 'Uploading...' : 'Upload transcript'}
            </button>
          </form>

          <aside className="upload-requirements surface" aria-labelledby="requirements-title">
            <h3 id="requirements-title">Before you upload</h3>
            <dl>
              <div><dt>File type</dt><dd>.txt</dd></div>
              <div><dt>Encoding</dt><dd>UTF-8</dd></div>
              <div><dt>Quantity</dt><dd>One transcript per upload</dd></div>
            </dl>
            <p>
              Uploading stores the transcript only. You will review it and start
              analysis from the transcript detail page.
            </p>
            <ol className="upload-next-steps">
              <li>Upload the transcript record.</li>
              <li>Open the record and start analysis.</li>
              <li>Review final signals and supporting evidence.</li>
            </ol>
          </aside>
        </div>
      )}
    </section>
  )
}
