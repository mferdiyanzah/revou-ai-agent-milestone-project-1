import type { NextApiRequest, NextApiResponse } from 'next'
import { PDFExtract } from 'pdf.js-extract'
import formidable from 'formidable'
import fs from 'fs'

type ResponseData = {
  data?: any;
  error?: string;
}

export const config = {
  api: {
    bodyParser: false,
  },
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseData>
) {
  const pdfExtract = new PDFExtract()
  const options = {} // default options

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    // Configure formidable with proper options
    const form = formidable({
      uploadDir: '/tmp', // or your preferred temp directory
      keepExtensions: true,
      maxFileSize: 10 * 1024 * 1024, // 10MB limit
    });

    const [fields, files] = await form.parse(req);
    
    console.log('Received fields:', fields);
    console.log('Received files:', files);

    // Check if file exists - handle both single file and array cases
    const uploadedFile = files.file;
    if (!uploadedFile) {
      console.error('No file field found in upload');
      return res.status(400).json({ error: 'No file uploaded - missing file field' });
    }

    // Handle both single file and array of files
    const file = Array.isArray(uploadedFile) ? uploadedFile[0] : uploadedFile;
    
    if (!file) {
      console.error('No file uploaded');
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const filePath = file.filepath;
    console.log('Processing file at path:', filePath);
    
    // Verify file exists on disk
    if (!fs.existsSync(filePath)) {
      console.error('File does not exist at path:', filePath);
      return res.status(400).json({ error: 'Uploaded file not found' });
    }

    // Log file details for debugging
    console.log('File details:', {
      originalFilename: file.originalFilename,
      mimetype: file.mimetype,
      size: file.size,
      filepath: filePath
    });

    try {
      const data = await pdfExtract.extract(filePath, options);
      
      // Check if any page has content
      const hasContent = data.pages.some(page => page.content.length > 0);
      if (!hasContent) {
        console.log('No text content found in PDF');
        return res.status(404).json({ error: 'No text content found in PDF' });
      }

      // Filter out empty strings and non-printable characters
      data.pages = data.pages.map(page => ({
        ...page,
        content: page.content.filter(item => item.str !== ' ' && !/[^\x20-\x7E]/.test(item.str))
      }));

      // Clean up the temporary file
      try {
        fs.unlinkSync(filePath);
      } catch (cleanupError) {
        console.warn('Failed to clean up temporary file:', cleanupError);
      }

      return res.status(200).json({ data: data });

    } catch (pdfError) {
      console.error('Error extracting PDF data:', pdfError);
      
      // Clean up the temporary file on error
      try {
        fs.unlinkSync(filePath);
      } catch (cleanupError) {
        console.warn('Failed to clean up temporary file after error:', cleanupError);
      }
      
      return res.status(500).json({ error: 'Failed to extract PDF data' });
    }

  } catch (error) {
    console.error('Error handling file upload:', error);
    res.status(500).json({ error: 'Failed to handle file upload' });
  }
} 