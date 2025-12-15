import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Configure PDF.js worker to use local file from public directory
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

interface PdfViewerProps {
  pdfUrl: string;
  onClose: () => void;
}

export default function PdfViewer({ pdfUrl, onClose }: PdfViewerProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageInput, setPageInput] = useState<string>("1");
  const [scale, setScale] = useState<number>(1.0);
  const [searchText, setSearchText] = useState<string>("");

  console.log('[PDF Viewer] Component mounted with URL:', pdfUrl);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        console.log('[PDF Viewer] ESC key pressed, closing viewer');
        onClose();
      }
    };
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  // Track scroll position to update current page
  useEffect(() => {
    const handleScroll = () => {
      if (!contentRef.current) return;
      const pages = contentRef.current.querySelectorAll('.react-pdf__Page');
      const scrollTop = contentRef.current.scrollTop;
      const offset = 100;

      for (let i = 0; i < pages.length; i++) {
        const page = pages[i] as HTMLElement;
        if (page.offsetTop - offset <= scrollTop && 
            page.offsetTop + page.offsetHeight - offset > scrollTop) {
          setCurrentPage(i + 1);
          setPageInput(String(i + 1));
          break;
        }
      }
    };

    const content = contentRef.current;
    if (content) {
      content.addEventListener('scroll', handleScroll);
      return () => content.removeEventListener('scroll', handleScroll);
    }
  }, [numPages]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === modalRef.current) {
      console.log('[PDF Viewer] Backdrop clicked, closing viewer');
      onClose();
    }
  };

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    console.log('[PDF Viewer] Document loaded successfully. Total pages:', numPages);
    setNumPages(numPages);
  };

  const onDocumentLoadError = (error: Error) => {
    console.error('[PDF Viewer] Error loading PDF:', error);
    console.error('[PDF Viewer] PDF URL was:', pdfUrl);
  };

  const handleDownload = async () => {
    console.log('[PDF Viewer] Download button clicked');
    console.log('[PDF Viewer] Fetching PDF from:', pdfUrl);
    try {
      const response = await fetch(pdfUrl);
      console.log('[PDF Viewer] Fetch response status:', response.status);
      const blob = await response.blob();
      console.log('[PDF Viewer] Blob created, size:', blob.size, 'bytes');
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "document.pdf";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      console.log('[PDF Viewer] Download initiated successfully');
    } catch (error) {
      console.error('[PDF Viewer] Download failed:', error);
    }
  };

  const scrollToPage = (pageNum: number) => {
    if (!contentRef.current || pageNum < 1 || pageNum > numPages) return;
    const pages = contentRef.current.querySelectorAll('.react-pdf__Page');
    const targetPage = pages[pageNum - 1] as HTMLElement;
    if (targetPage) {
      targetPage.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const goToPrevPage = () => {
    const newPage = Math.max(currentPage - 1, 1);
    console.log('[PDF Viewer] Navigate to previous page:', newPage);
    scrollToPage(newPage);
  };

  const goToNextPage = () => {
    const newPage = Math.min(currentPage + 1, numPages);
    console.log('[PDF Viewer] Navigate to next page:', newPage);
    scrollToPage(newPage);
  };

  const handlePageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPageInput(e.target.value);
  };

  const handlePageInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const pageNum = parseInt(pageInput, 10);
    if (!isNaN(pageNum) && pageNum >= 1 && pageNum <= numPages) {
      scrollToPage(pageNum);
    } else {
      setPageInput(String(currentPage));
    }
  };

  const handleSearch = () => {
    if (!searchText || !contentRef.current) return;
    
    // Use browser's native find functionality
    if (window.find) {
      window.find(searchText, false, false, true, false, true, false);
    }
  };

  const handleSearchKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const zoomIn = () => {
    const newScale = Math.min(scale + 0.2, 3.0);
    console.log('[PDF Viewer] Zoom in to:', Math.round(newScale * 100) + '%');
    setScale(newScale);
  };

  const zoomOut = () => {
    const newScale = Math.max(scale - 0.2, 0.5);
    console.log('[PDF Viewer] Zoom out to:', Math.round(newScale * 100) + '%');
    setScale(newScale);
  };

  return (
    <div className="pdf-modal-backdrop" ref={modalRef} onClick={handleBackdropClick}>
      <div className="pdf-modal-container">
        <div className="pdf-modal-header">
          <h3>PDF Viewer</h3>
          <div className="pdf-modal-actions">
            <div className="pdf-search">
              <input
                type="text"
                placeholder="Find in document..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                onKeyPress={handleSearchKeyPress}
                className="pdf-search-input"
              />
              <button className="pdf-search-button" onClick={handleSearch} title="Find">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
              </button>
            </div>
            <div className="pdf-controls">
              <button className="pdf-control-button" onClick={zoomOut} title="Zoom Out">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="8" y1="11" x2="14" y2="11"></line>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
              </button>
              <span className="zoom-level">{Math.round(scale * 100)}%</span>
              <button className="pdf-control-button" onClick={zoomIn} title="Zoom In">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="11" y1="8" x2="11" y2="14"></line>
                  <line x1="8" y1="11" x2="14" y2="11"></line>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
              </button>
            </div>
            <div className="pdf-pagination">
              <button className="pdf-control-button" onClick={goToPrevPage} disabled={currentPage <= 1}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="15 18 9 12 15 6"></polyline>
                </svg>
              </button>
              <form onSubmit={handlePageInputSubmit} className="page-input-form">
                <input
                  type="text"
                  value={pageInput}
                  onChange={handlePageInputChange}
                  className="page-input"
                  title="Page number"
                />
                <span className="page-separator">/</span>
                <span className="page-total">{numPages || "?"}</span>
              </form>
              <button className="pdf-control-button" onClick={goToNextPage} disabled={currentPage >= numPages}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="9 18 15 12 9 6"></polyline>
                </svg>
              </button>
            </div>
            <button className="pdf-download-button" onClick={handleDownload}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              Download
            </button>
            <button className="pdf-close-button" onClick={onClose}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        </div>
        <div className="pdf-modal-content" ref={contentRef}>
          <div className="pdf-document-container">
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={<div className="pdf-loading">Loading PDF...</div>}
              error={<div className="pdf-error">Failed to load PDF. Please try downloading instead.</div>}
            >
              {Array.from(new Array(numPages), (_, index) => (
                <Page
                  key={`page_${index + 1}`}
                  pageNumber={index + 1}
                  scale={scale}
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                  className="pdf-page"
                />
              ))}
            </Document>
          </div>
        </div>
      </div>
    </div>
  );
}
